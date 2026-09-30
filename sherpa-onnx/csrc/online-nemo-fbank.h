// sherpa-onnx/csrc/online-nemo-fbank.h
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: the streaming half of the feature extractor's NeMo mode
// (FeatureExtractorConfig::is_nemo). It turns audio arriving in chunks into
// NeMo's log-mel frames, computed by the shared NeMo frontend
// (nemo-frontend.h).
//
// The feature: live speech recognition with Nemotron, a streaming NeMo
// transducer. Nemotron was trained on NeMo's features. The
// default kaldi-native-fbank path computes them differently: a higher log
// floor, frames centred half a hop (80 samples) later, per-frame pre-emphasis
// and a periodic window. The streaming NeMo transducer recognizer therefore
// selects this mode, and every other feature path in the library stays on its
// own extractor. The fix is a separate extractor, not an edit of the shared
// kaldi-native-fbank code, because that code also computes the features of
// every other model, the speaker fingerprint extractors included.
//
// Framing
//
// Frame f is centred on sample f * hop (160) and reads samples
// [f*hop - 256, f*hop + 256); samples before the signal and after its end read
// as zero. So:
//
//  - while input arrives, frame f is ready once samples up to f*hop + 255 have
//    arrived;
//  - at InputFinished the total is NeMo's valid frame count, n // hop for n
//    samples (FilterbankFeatures.get_seq_len with center=True). NeMo computes
//    one more centred frame, 1 + n // hop, and overwrites it with its pad
//    value, so that frame is padding and is never produced here;
//  - no leading silent samples are needed to line the grid up with NeMo's.
//
// Pre-emphasis runs over the whole stream, y[0] = x[0] and
// y[i] = x[i] - 0.97 * x[i-1], carrying the previous raw sample across chunks.
// Pre-emphasising each chunk on its own would put a discontinuity at every
// chunk boundary. NemoPreemphasisBuffer (nemo-frontend.h) does it and holds
// the samples, the same class the Streaming Sortformer engine uses.
//
// No normalization is applied, as Nemotron's preprocessor sets none, and no
// dither.
//
// Memory
//
// A stream can run for hours, so nothing grows with its length. Pre-emphasised
// samples are dropped as soon as no uncomputed frame can read them, and frames
// are dropped when the caller pops them, as knf::OnlineFbank does.
// NumFramesHeld and NumSamplesHeld expose both counts for the tests.
//
// Indexing matches knf::OnlineFbank: frame indices are absolute from the start
// of the stream, NumFramesReady counts every frame computed so far, popped or
// not, and Pop(n) drops the n oldest held frames.
//
// Not thread-safe; FeatureExtractor serialises the calls.

#ifndef SHERPA_ONNX_CSRC_ONLINE_NEMO_FBANK_H_
#define SHERPA_ONNX_CSRC_ONLINE_NEMO_FBANK_H_

#include <cstdint>
#include <vector>

#include "sherpa-onnx/csrc/nemo-frontend.h"

namespace sherpa_onnx {

class OnlineNemoFbank {
 public:
  explicit OnlineNemoFbank(const NemoFrontendConfig &config);

  OnlineNemoFbank(const OnlineNemoFbank &) = delete;
  OnlineNemoFbank &operator=(const OnlineNemoFbank &) = delete;

  // `sampling_rate` must equal config.sample_rate; FeatureExtractor resamples
  // before calling this. `waveform` is in [-1, 1].
  void AcceptWaveform(float sampling_rate, const float *waveform, int32_t n);

  // Computes the remaining frames up to n // hop. Further calls do nothing.
  void InputFinished();

  int32_t NumFramesReady() const;

  // True only after InputFinished, for the last frame.
  bool IsLastFrame(int32_t frame) const;

  // `frame` is absolute and must be held: popped frames are gone.
  const float *GetFrame(int32_t frame) const;

  // Drops the n oldest held frames.
  void Pop(int32_t n);

  int32_t Dim() const;

  int32_t NumFramesHeld() const;
  int64_t NumSamplesHeld() const;

 private:
  void ComputeReadyFrames();

  NemoFrontendConfig config_;
  NemoFrontend frontend_;
  NemoPreemphasisBuffer preemph_buffer_;

  // Frames covering absolute indices [frames_start_, num_frames_), time-major.
  std::vector<float> frames_;
  int64_t frames_start_ = 0;
  int64_t num_frames_ = 0;

  bool input_finished_ = false;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_NEMO_FBANK_H_
