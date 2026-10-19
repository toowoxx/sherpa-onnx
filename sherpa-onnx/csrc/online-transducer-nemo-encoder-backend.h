// sherpa-onnx/csrc/online-transducer-nemo-encoder-backend.h
//
// The encoder of the streaming NeMo transducer (Nemotron) run by the
// caller instead of ONNX Runtime. A platform with a faster runtime for the same
// graph than ONNX Runtime on the CPU (CoreML on Apple devices, which can use
// the GPU and the Neural Engine) supplies the encoder as a function. The
// decoder, the joiner, the feature extraction and the decoding loop stay in
// this library, so the only thing the platform replaces is the encoder.
//
// The function is the pure function of the exported encoder graph for one
// stream: one window of features and the three cache tensors in, the encoded
// frames and the next caches out. The library calls it once per stream and
// window. The tensors are float32 and row-major, with the layout of the ONNX
// graph without its batch axis:
//   features             (n_frames, feature_dim)      (the graph takes the
//                                                      transpose, the callee
//                                                      does that)
//   cache_last_channel   (layers, channel_time, dim)
//   cache_last_time      (layers, dim, time_width)
//   encoded              (encoded_dim, encoded_frames)
//
// The values the library reads from the ONNX metadata of the encoder
// (window_size, chunk_shift, the cache shapes, the prompt dictionary, ...) come
// from `metadata`, which answers the same keys. The library does not own the
// resources behind the two functions: whoever creates the backend releases
// them after the recognizer is destroyed.

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_ENCODER_BACKEND_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_ENCODER_BACKEND_H_

#include <cstdint>
#include <functional>
#include <string>

namespace sherpa_onnx {

// What one call of the external encoder is given.
struct NemoEncoderRequest {
  const float *features = nullptr;  // (n_frames, feature_dim)
  int32_t n_frames = 0;
  int32_t feature_dim = 0;
  const float *cache_last_channel = nullptr;
  int64_t cache_last_channel_count = 0;  // number of floats
  const float *cache_last_time = nullptr;
  int64_t cache_last_time_count = 0;  // number of floats
  int64_t cache_last_channel_len = 0;
  int64_t prompt_id = 0;  // only read by a multilingual encoder
};

// Buffers the external encoder writes into. The cache buffers have the sizes
// of the request's caches. `encoded` holds `encoded_capacity` floats and the
// callee sets the number of frames it wrote.
struct NemoEncoderResult {
  float *encoded = nullptr;  // (encoded_dim, encoded_frames)
  int64_t encoded_capacity = 0;
  int32_t encoded_frames = 0;
  float *cache_last_channel = nullptr;
  float *cache_last_time = nullptr;
  int64_t cache_last_channel_len = 0;
};

// Runs the encoder for one stream. Returns true on success.
using NemoEncoderRunFn =
    std::function<bool(const NemoEncoderRequest &, NemoEncoderResult *)>;

// Returns the value of a key of the ONNX metadata of the encoder, or an empty
// string if the key does not exist.
using NemoEncoderMetadataFn = std::function<std::string(const std::string &)>;

struct NemoEncoderExternalBackend {
  NemoEncoderRunFn run;
  NemoEncoderMetadataFn metadata;
  int32_t feature_dim = 0;
  // Number of channels of the encoded frames.
  int32_t encoded_dim = 0;
  // True if the encoder takes a language prompt.
  bool multilingual = false;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_ENCODER_BACKEND_H_
