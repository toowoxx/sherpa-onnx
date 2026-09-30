// sherpa-onnx/csrc/nemo-frontend.h
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: the one NeMo-exact log-mel implementation in this fork, reproducing
// NeMo's AudioToMelSpectrogramPreprocessor closely enough that an exported
// NeMo graph sees what PyTorch saw.
//
// The features: speaker diarization and speech recognition with NeMo-trained
// models. Every model here that was trained on NeMo's log-mel features reads
// them from this file:
//
//  - the Streaming Sortformer diarization engine (streaming-sortformer-
//    diarization.cc), which runs ComputeFrameRange with its own framing;
//  - the feature extractor's NeMo mode (features.cc, is_nemo), streamed by
//    OnlineNemoFbank (online-nemo-fbank.h) with NeMo's own frame count.
//
// Raw 16 kHz mono audio in, 128-dimensional log-mel frames out. How closely
// these frames match NeMo's decides how accurate the model's output is: see
// the measurement below.
//
// Why one implementation: every caller must match NeMo bit for bit, and two
// copies of a computation held to that standard drift apart. A caller that
// needs different framing (the frame count, when a frame is ready, what happens
// at the end of the input) builds it around ComputeFrameRange; it does not copy
// the chain. The streaming callers also share NemoPreemphasisBuffer, which
// carries step 1 across calls and drops the samples no uncomputed frame reads.
//
// Why this is written from scratch instead of configured
//
// The obvious implementation is kaldi-native-fbank, which this repository
// already vendors and which has a Slaney/librosa mel mode. That was measured
// and rejected. At its best configuration (global preemphasis, snip_edges with
// a 200-sample prepend, is_librosa, slaney norm, power spectrum) its frames
// reach a per-frame feature cosine very close to 1 against NeMo, and still
// cost real accuracy end to end: the Sortformer engine agrees with the NeMo
// reference on fewer frames with them than with the chain implemented here.
//
// The lesson worth keeping: feature cosine is not a proxy for diarization
// agreement. The features feed 35 encoder layers and then a speaker cache that
// carries state forward, so a perturbation far below any reasonable feature
// tolerance can flip a cache-compression decision and change every later chunk.
// "The cosine is nearly 1, ship it" is exactly the reasoning that loses those
// frames. Do not re-derive this by swapping the frontend back.
//
// The chain, in order — every step is load-bearing
//
//  1. Global pre-emphasis over the WHOLE signal before framing:
//     y[0] = x[0], y[i] = x[i] - 0.97*x[i-1]. Kaldi pre-emphasises per frame
//     using the frame's own first sample as the predecessor, which differs at
//     every frame boundary.
//  2. ZERO-pad n_fft/2 (=256) samples at both ends. This is what
//     torch.stft(center=True, pad_mode="constant") does, and it centres frame t
//     on sample t*hop. Kaldi's snip_edges=true starts frame t at t*hop; the
//     two conventions differ by half a window.
//
//     Not REFLECT padding, even though reflect is torch.stft's own default.
//     Against what NeMo actually produced, reflect puts the first frame
//     several units out in the log domain and the final frame out as well,
//     while zeros agree with NeMo across the whole clip. Reflect makes the
//     first and last frames see mirrored signal where NeMo sees silence.
//  3. Frame at hop_length, window length win_length (400).
//  4. Window with a SYMMETRIC Hann of length 400 — hann_window(periodic=False),
//     denominator win_length-1, not win_length. Then zero-pad it symmetrically
//     into n_fft (56 zeros each side). torch.stft pads the window, not the
//     signal frame, and centred padding is not the same as right padding.
//  5. Power spectrum: real^2 + imag^2 (mag_power = 2.0), 257 bins.
//  6. Librosa Slaney mel filterbank: 128 bins, 0-8000 Hz, htk=False,
//     norm="slaney".
//  7. log(x + 2^-24). NeMo's log_zero_guard_value in "add" mode. Using a clamp
//     instead of an addition changes digital-silence bins by several units.
//
// Audio is NOT peak-normalised and NOT dithered. NeMo's process_signal skips
// normalisation when streaming_mode is set, and dither is zero in eval. Adding
// either shifts every sigmoid.
//
// Output layout is time-major: frame t occupies [t*n_mels, (t+1)*n_mels). The
// exported graph's `chunk` input is (1, frames, 128), so this is the layout it
// consumes directly.

#ifndef SHERPA_ONNX_CSRC_NEMO_FRONTEND_H_
#define SHERPA_ONNX_CSRC_NEMO_FRONTEND_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {

// These are not free parameters: they reproduce
// AudioToMelSpectrogramPreprocessor as configured in both the
// diar_streaming_sortformer_4spk-v2.1 checkpoint and
// Nemotron's preprocessor. See the SPEC above for why a Kaldi-configured
// frontend is not an acceptable substitute.
struct NemoFrontendConfig {
  int32_t sample_rate = 16000;
  int32_t n_fft = 512;
  int32_t win_length = 400;
  int32_t hop_length = 160;
  int32_t n_mels = 128;
  float preemph = 0.97f;
  // 2^-24, NeMo's log_zero_guard_value in "add" mode.
  float log_zero_guard = 5.960464477539063e-08f;
  float low_freq = 0.0f;
  float high_freq = 8000.0f;

  bool Validate() const;
  std::string ToString() const;
};

class NemoFrontend {
 public:
  explicit NemoFrontend(const NemoFrontendConfig &config);
  ~NemoFrontend();

  NemoFrontend(const NemoFrontend &) = delete;
  NemoFrontend &operator=(const NemoFrontend &) = delete;

  // Number of log-mel frames produced for `n_samples` input samples. This is
  // torch.stft's centred-frame count, 1 + n_samples / hop_length.
  int32_t NumFrames(int32_t n_samples) const;

  // Computes log-mel features for a whole clip. `samples` is 16 kHz mono in
  // [-1, 1]. The result is time-major, NumFrames(samples.size()) * n_mels
  // floats.
  //
  // The convenience form used by the tests; the streaming callers run
  // ComputeFrameRange directly so they never hold a whole stream in memory.
  std::vector<float> Compute(const std::vector<float> &samples) const;

  // Steps 2-7 for frames [first_frame, first_frame + n_frames), reading the
  // ALREADY PRE-EMPHASISED signal from `window`, which covers absolute sample
  // indices [window_start, window_start + window.size()).
  //
  // Samples outside that range read as zero, which is exactly the padding
  // convention, so a caller streaming audio in never builds a padded buffer and
  // never needs to know the clip length in advance. Pre-emphasis happens
  // before, in NemoPreemphasisBuffer for the streaming callers, because it is
  // a one-sample recurrence: splitting it here would make every window
  // boundary a discontinuity.
  //
  // Frame f is centred on sample f * hop_length and spans
  // [f*hop - n_fft/2, f*hop + n_fft/2), so a caller can compute frame f as soon
  // as it holds samples up to f*hop + n_fft/2 - 1.
  std::vector<float> ComputeFrameRange(const std::vector<float> &window,
                                       int64_t window_start,
                                       int64_t first_frame,
                                       int32_t n_frames) const;

  // The mel filterbank, exposed for inspection in tests. Row-major,
  // n_mels * (n_fft/2 + 1).
  const std::vector<float> &MelFilters() const;

  int32_t NumMels() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Step 1 of the chain for a stream, and the samples ComputeFrameRange reads.
// Both streaming callers hold their audio here: the feature extractor's NeMo
// mode (OnlineNemoFbank) and the Streaming Sortformer engine. Each keeps its
// own frame count and decides itself which frames are ready; this class
// pre-emphasises the stream across calls and drops the samples no uncomputed
// frame reads.
//
// The stream's first sample passes through, and every later sample subtracts
// preemph times its raw predecessor, the last raw sample of the previous call
// included. Pre-emphasising each call on its own would put a discontinuity at
// every call boundary. The arithmetic is the one Compute uses, so a streamed
// frame equals the whole-clip frame bit for bit.
class NemoPreemphasisBuffer {
 public:
  explicit NemoPreemphasisBuffer(const NemoFrontendConfig &config);

  // Pre-emphasises `n` raw samples and appends them.
  void Accept(const float *samples, int32_t n);

  // Drops every held sample that no frame from `next_frame` on reads. Frame f
  // reads from sample f * hop_length - n_fft / 2.
  void DropBeforeFrame(int64_t next_frame);

  // Returns to an empty stream.
  void Reset();

  // The held pre-emphasised samples, covering absolute sample indices
  // [Start(), Start() + Samples().size()): the `window` and `window_start` of
  // ComputeFrameRange.
  const std::vector<float> &Samples() const { return samples_; }
  int64_t Start() const { return start_; }

  // Every sample accepted since the stream began or was reset.
  int64_t NumAccepted() const { return num_accepted_; }

 private:
  float preemph_;
  int32_t hop_length_;
  int32_t half_fft_;

  std::vector<float> samples_;
  int64_t start_ = 0;
  int64_t num_accepted_ = 0;
  float last_raw_sample_ = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_NEMO_FRONTEND_H_
