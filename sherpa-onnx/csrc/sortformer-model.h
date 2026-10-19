// sherpa-onnx/csrc/sortformer-model.h
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: the ONNX Runtime session around the exported Streaming Sortformer v2.1
// graph — the neural half of the diarization engine.
//
// The feature: speaker diarization, which tells who spoke when in a
// recording. This wraps NVIDIA's diar_streaming_sortformer_4spk-v2.1
// checkpoint, exported to ONNX opset 17.
//
// The graph is a PURE FUNCTION of (chunk features, speaker cache, FIFO). It
// holds no state between calls; the streaming behaviour lives entirely in
// SortformerAosc. Keeping the two apart is what makes the AOSC testable on
// recorded per-chunk tensors without a 492 MB model in the loop.
//
// I/O contract of the export
//
//   inputs
//     chunk             (1, chunk_feat_frames, 128) float32  log-mel
//     chunk_lengths     (1,) int64
//     spkcache          (1, spkcache_capacity, 512) float32
//     spkcache_lengths  (1,) int64
//     fifo              (1, fifo_capacity, 512) float32
//     fifo_lengths      (1,) int64
//   outputs
//     spkcache_fifo_chunk_preds  (1, out_frames, 4) float32
//     chunk_pre_encode_embs      (1, chunk_enc_frames, 512) float32
//     chunk_pre_encode_lengths   (1,) int64
//
// Every time axis is DYNAMIC, and that is load-bearing rather than a
// convenience: the first chunk has no left context and the last is truncated,
// so a fixed-shape export cannot run either of them (verified — a static export
// fails exactly those two cases). The dynamic export also means one artifact
// serves both the 10.0s and the 30.4s latency configurations.
//
// What the obvious-looking refactor breaks
//
//  * The cache and FIFO tensors are passed at FULL capacity with zero padding
//    beyond their valid lengths, never tightly packed. Padding is provably free
//    here (measured max abs diff 0.0 against a tight concat), but the graph's
//    internal concat_and_pad LEFT-PACKS the three regions — valid cache, then
//    valid FIFO, then the chunk, with all padding at the end. Feeding tight
//    tensors changes nothing; feeding a naive three-way concatenation of
//    capacity-sized tensors interleaves padding into the middle and the
//    prediction gather indices then point at the wrong frames.
//  * Output names must be looked up, not assumed positional. The export names
//    two of the three outputs after internal ops
//    (`Mulspkcache_fifo_chunk_preds_dim_0`), which is a strong hint that the
//    ordering is an export artifact rather than a contract.
//
// Only the CPU execution provider is supported by the ONNX Runtime backend.
//
// External backend
//
// A platform with a faster runtime for the same graph than ONNX Runtime on the
// CPU (CoreML on Apple devices, which can use the GPU and the Neural Engine)
// supplies the forward step as a function through SortformerExternalBackend.
// The step is the same pure function: this library keeps the frontend, the
// AOSC and the post-processing, so the only thing the platform replaces is the
// neural network. A fixed-shape network is served by padding on the callee's
// side with the lengths this class passes.

#ifndef SHERPA_ONNX_CSRC_SORTFORMER_MODEL_H_
#define SHERPA_ONNX_CSRC_SORTFORMER_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/sortformer-diarization-config.h"

namespace sherpa_onnx {

struct SortformerModelOutput {
  // (n_packed, n_spk) row-major.
  std::vector<float> preds;
  int32_t n_packed = 0;

  // (n_chunk_enc, emb_dim) row-major.
  std::vector<float> chunk_pre_encode_embs;
  int32_t n_chunk_enc = 0;

  int32_t chunk_pre_encode_length = 0;
};

// What one call of the external forward step is given. The state tensors are
// at full capacity with zero padding behind their valid lengths, as for the
// ONNX Runtime backend.
struct SortformerForwardRequest {
  const float *features = nullptr;  // (n_feat_frames, feat_dim)
  int32_t n_feat_frames = 0;
  const float *spkcache = nullptr;  // (spkcache_capacity, emb_dim)
  int32_t spkcache_capacity = 0;
  int32_t spkcache_length = 0;
  const float *fifo = nullptr;  // (fifo_capacity, emb_dim)
  int32_t fifo_capacity = 0;
  int32_t fifo_length = 0;
};

// Buffers the external forward step writes into. The caller sizes them for the
// largest result the geometry allows and the callee sets the counts.
struct SortformerForwardResult {
  float *preds = nullptr;  // (preds_capacity, n_spk)
  int32_t preds_capacity = 0;
  int32_t n_packed = 0;
  float *chunk_pre_encode_embs = nullptr;  // (embs_capacity, emb_dim)
  int32_t embs_capacity = 0;
  int32_t n_chunk_enc = 0;
  int32_t chunk_pre_encode_length = 0;
};

// Returns true on success.
using SortformerExternalForwardFn = bool (*)(void *user_data,
                                             const SortformerForwardRequest *,
                                             SortformerForwardResult *);

struct SortformerExternalBackend {
  void *user_data = nullptr;
  SortformerExternalForwardFn forward = nullptr;
  // Called once when the model is destroyed. May be null.
  void (*destroy)(void *user_data) = nullptr;
  // Identity of the artifact the backend runs, reported as the model SHA.
  std::string model_id;
};

class SortformerModel {
 public:
  explicit SortformerModel(const SortformerDiarizationConfig &config);

  SortformerModel(const SortformerDiarizationConfig &config,
                  SortformerExternalBackend backend);

  template <typename Manager>
  SortformerModel(Manager *mgr, const SortformerDiarizationConfig &config);

  ~SortformerModel();

  SortformerModel(const SortformerModel &) = delete;
  SortformerModel &operator=(const SortformerModel &) = delete;

  // `features` is (n_feat_frames, feat_dim) row-major log-mel.
  // `spkcache` is (spkcache_len, emb_dim) and `fifo` is (fifo_len, emb_dim),
  // both at full capacity.
  SortformerModelOutput Forward(const float *features, int32_t n_feat_frames,
                                const std::vector<float> &spkcache,
                                int32_t spkcache_lengths,
                                const std::vector<float> &fifo,
                                int32_t fifo_lengths) const;

  // SHA-256 of the model file, lowercase hex. It proves which model file a
  // session ran; a log line naming the path is not enough, because one path
  // can hold different model files over time.
  const std::string &ModelSha256() const;

  // ONNX Runtime intra-op worker thread ids, so a caller can hand the threads
  // that run inference to a platform scheduling API. Empty when the runtime
  // did not report them. This class only records the ids and never acts on
  // them.
  const std::vector<int32_t> &WorkerThreadIds() const;

 public:
  // The backends derive from this; it is not part of the interface.
  class Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SORTFORMER_MODEL_H_
