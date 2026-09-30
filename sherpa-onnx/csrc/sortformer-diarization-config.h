// sherpa-onnx/csrc/sortformer-diarization-config.h
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: Streaming Sortformer diarization — the one place every magic number of
// the feature lives.
//
// The feature: speaker diarization, which tells who spoke when in a
// recording. The library's other diarizer clusters speaker embeddings, and
// clustering can collapse several distinct speakers into one. NVIDIA's
// Streaming Sortformer v2.1 is an end-to-end neural diarizer that assigns
// frames to at most four "slots" directly, with no clustering step to
// collapse. This config carries its constants.
//
// Why the numbers live together: the streaming geometry, the AOSC constants and
// the post-processing thresholds are not independently tunable. The geometry
// numbers reproduce NVIDIA's published 10.0s-latency configuration, the AOSC
// numbers are the defaults the exported checkpoint was validated against, and
// the post-processing numbers are the CallHome-tuned set shipped with the
// checkpoint. Changing one in isolation means the engine no longer computes
// what NeMo computes with the published checkpoint, and every comparison of
// the engine against NeMo stops holding.
//
// What the obvious refactor breaks
//
// `spkcache_len_per_spk` is `spkcache_len / n_spk - sil_frames_per_spk`, i.e.
// integer division FIRST. 188/4 - 3 = 44. Computing it as
// (188 - 4*3)/4 = 44 agrees here and disagrees for other geometries; NeMo does
// the former and the boost counts derived from it decide which frames survive
// cache compression.
//
// The three boost counts each floor a rate times that per-speaker capacity.
// `weak_boost_rate` is 1.5 — GREATER than one, so `weak_boost_per_spk` (66) is
// larger than the per-speaker capacity (44). That is not a bug to "fix": the
// weak boost deliberately reaches past what one speaker can keep, because it
// runs after the strong boost and on the strong boost's output, and its job is
// to rank the remainder rather than to reserve capacity.
//
// Post-processing has exactly SIX parameters and NO median filter. Upstream
// diar_streaming_sortformer_4spk-v2_callhome-part1.yaml has no such parameter
// and NeMo's predlist_to_timestamps applies only hysteresis binarization,
// padding and duration filtering. Adding a median filter would silently move
// every segment boundary away from the segments NeMo produces.
//
// The one exception to "every magic number lives here": the log-mel frontend's
// constants are NemoFrontendConfig in nemo-frontend.h, next to the shared NeMo
// frontend that other NeMo-trained models read as well. `frontend` below holds
// them for this engine.

#ifndef SHERPA_ONNX_CSRC_SORTFORMER_DIARIZATION_CONFIG_H_
#define SHERPA_ONNX_CSRC_SORTFORMER_DIARIZATION_CONFIG_H_

#include <cmath>
#include <cstdint>
#include <string>

#include "sherpa-onnx/csrc/nemo-frontend.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// Streaming geometry plus the Arrival-Order Speaker Cache constants. Defaults
// are the 10.0s-latency configuration (124/1/124/188).
struct SortformerStreamingConfig {
  // Encoder frames of new audio per chunk.
  int32_t chunk_len = 124;
  // Encoder frames of context fed to the graph but excluded from the state
  // update. They exist to give the encoder receptive field at the chunk edges;
  // writing them into the FIFO double-counts every frame.
  int32_t chunk_left_context = 1;
  int32_t chunk_right_context = 1;
  // Capacity of the FIFO holding frames not yet committed to the cache.
  int32_t fifo_len = 124;
  // Capacity of the speaker cache.
  int32_t spkcache_len = 188;
  // Nominal number of frames moved out of the FIFO per update.
  int32_t spkcache_update_period = 144;
  // Encoder subsampling: one output frame per 8 feature frames (80 ms).
  int32_t subsampling_factor = 8;
  int32_t n_spk = 4;
  int32_t emb_dim = 512;
  int32_t feat_dim = 128;

  // Cache slots per speaker reserved for the running silence profile.
  int32_t spkcache_sil_frames_per_spk = 3;
  // Posterior clamp used when turning posteriors into cache-importance scores.
  float pred_score_threshold = 0.25f;
  // A popped frame counts as silence when its posteriors sum below this.
  float sil_threshold = 0.2f;
  float strong_boost_rate = 0.75f;
  float weak_boost_rate = 1.5f;
  float min_pos_scores_rate = 0.5f;
  // Recency prior added to the newly appended region of the candidate buffer.
  float scores_boost_latest = 0.05f;
  // Sentinel marking a cache slot as disabled during compression. Must exceed
  // any reachable flat index (n_spk * candidate width).
  int32_t max_index = 99999;

  int32_t SpkcacheLenPerSpk() const {
    return spkcache_len / n_spk - spkcache_sil_frames_per_spk;
  }
  int32_t StrongBoostPerSpk() const {
    return static_cast<int32_t>(
        std::floor(SpkcacheLenPerSpk() * strong_boost_rate));
  }
  int32_t WeakBoostPerSpk() const {
    return static_cast<int32_t>(
        std::floor(SpkcacheLenPerSpk() * weak_boost_rate));
  }
  int32_t MinPosScoresPerSpk() const {
    return static_cast<int32_t>(
        std::floor(SpkcacheLenPerSpk() * min_pos_scores_rate));
  }

  // Seconds of audio per encoder output frame.
  float FrameDurationSeconds() const {
    return static_cast<float>(subsampling_factor) * 160.0f / 16000.0f;
  }

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

// NeMo's CallHome-tuned post-processing, exactly six parameters. Seconds.
struct SortformerPostProcessingConfig {
  // Hysteresis: a speaker turns on above `onset` and stays on until `offset`.
  float onset = 0.641f;
  float offset = 0.561f;
  // Each detected region is widened by this much on each side before merging.
  float pad_onset = 0.229f;
  float pad_offset = 0.079f;
  // Segments shorter than this are dropped; gaps shorter than this are filled.
  float min_duration_on = 0.511f;
  float min_duration_off = 0.296f;

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

struct SortformerDiarizationConfig {
  std::string model;
  SortformerStreamingConfig streaming;
  NemoFrontendConfig frontend;
  SortformerPostProcessingConfig post_processing;

  int32_t num_threads = 2;
  bool debug = false;
  std::string provider = "cpu";

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SORTFORMER_DIARIZATION_CONFIG_H_
