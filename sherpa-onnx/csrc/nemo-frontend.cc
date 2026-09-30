// sherpa-onnx/csrc/nemo-frontend.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: see nemo-frontend.h for the features it serves, the measurement that
// excluded kaldi-native-fbank, and the ordered chain this file implements.

#include "sherpa-onnx/csrc/nemo-frontend.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/rfft.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

// Slaney (librosa htk=False) hz <-> mel. Linear below 1 kHz, logarithmic above.
constexpr float kMelFSp = 200.0f / 3.0f;
constexpr float kMelMinLogHz = 1000.0f;
constexpr float kMelMinLogMel = kMelMinLogHz / kMelFSp;  // 15.0

float HzToMelSlaney(float hz) {
  if (hz < kMelMinLogHz) {
    return hz / kMelFSp;
  }
  const float logstep = std::log(6.4f) / 27.0f;
  return kMelMinLogMel + std::log(hz / kMelMinLogHz) / logstep;
}

float MelToHzSlaney(float mel) {
  if (mel < kMelMinLogMel) {
    return kMelFSp * mel;
  }
  const float logstep = std::log(6.4f) / 27.0f;
  return kMelMinLogHz * std::exp(logstep * (mel - kMelMinLogMel));
}

// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney").
// Row-major n_mels x (n_fft/2 + 1).
std::vector<float> BuildSlaneyMelFilters(int32_t sample_rate, int32_t n_fft,
                                         int32_t n_mels, float low_freq,
                                         float high_freq) {
  const int32_t n_bins = n_fft / 2 + 1;

  std::vector<float> fft_freqs(n_bins);
  for (int32_t k = 0; k != n_bins; ++k) {
    fft_freqs[k] = static_cast<float>(k) * sample_rate / n_fft;
  }

  // n_mels + 2 band edges, equally spaced on the mel scale.
  std::vector<float> mel_f(n_mels + 2);
  {
    const float min_mel = HzToMelSlaney(low_freq);
    const float max_mel = HzToMelSlaney(high_freq);
    for (int32_t i = 0; i != n_mels + 2; ++i) {
      const float mel = min_mel + (max_mel - min_mel) * i / (n_mels + 1);
      mel_f[i] = MelToHzSlaney(mel);
    }
  }

  std::vector<float> weights(static_cast<size_t>(n_mels) * n_bins, 0.0f);
  for (int32_t m = 0; m != n_mels; ++m) {
    const float f_left = mel_f[m];
    const float f_center = mel_f[m + 1];
    const float f_right = mel_f[m + 2];
    const float fdiff_lower = f_center - f_left;
    const float fdiff_upper = f_right - f_center;

    for (int32_t k = 0; k != n_bins; ++k) {
      const float lower = (fft_freqs[k] - f_left) / fdiff_lower;
      const float upper = (f_right - fft_freqs[k]) / fdiff_upper;
      const float w = std::max(0.0f, std::min(lower, upper));
      // norm="slaney": each band is scaled to unit AREA rather than unit peak,
      // so wide high-frequency bands do not dominate the log-mel. Dropping this
      // is a silent, monotonically increasing bias across the spectrum.
      weights[static_cast<size_t>(m) * n_bins + k] = w * 2.0f / (f_right - f_left);
    }
  }

  return weights;
}

// hann_window(win_length, periodic=False): denominator win_length - 1. The
// periodic variant (denominator win_length) is what most FFT code uses and what
// torch.hann_window defaults to; NeMo passes periodic=False.
std::vector<float> BuildSymmetricHann(int32_t win_length) {
  std::vector<float> w(win_length);
  if (win_length == 1) {
    w[0] = 1.0f;
    return w;
  }
  // Not M_PI: it is not in the C++ standard and needs _USE_MATH_DEFINES on
  // MSVC, which this file would then have to define before every include.
  constexpr double kPi = 3.14159265358979323846;
  for (int32_t i = 0; i != win_length; ++i) {
    w[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i /
                                                   (win_length - 1)));
  }
  return w;
}

}  // namespace

class NemoFrontend::Impl {
 public:
  explicit Impl(const NemoFrontendConfig &config)
      : config_(config),
        rfft_(config.n_fft),
        mel_filters_(BuildSlaneyMelFilters(config.sample_rate, config.n_fft,
                                           config.n_mels, config.low_freq,
                                           config.high_freq)) {
    // torch.stft zero-pads the WINDOW symmetrically into n_fft, so the padded
    // window is centred in the FFT buffer. Precomputed here at full n_fft width
    // so the inner loop is a plain elementwise multiply.
    const int32_t pad = (config_.n_fft - config_.win_length) / 2;
    padded_window_.assign(config_.n_fft, 0.0f);
    const std::vector<float> hann = BuildSymmetricHann(config_.win_length);
    std::copy(hann.begin(), hann.end(), padded_window_.begin() + pad);
  }

  int32_t NumFrames(int32_t n_samples) const {
    if (n_samples < 0) {
      return 0;
    }
    return 1 + n_samples / config_.hop_length;
  }

  std::vector<float> Compute(const std::vector<float> &samples) const {
    const int32_t n_samples = static_cast<int32_t>(samples.size());

    // Step 1: global pre-emphasis, over the whole signal, before any framing.
    std::vector<float> pre(n_samples);
    if (n_samples > 0) {
      pre[0] = samples[0];
      for (int32_t i = 1; i < n_samples; ++i) {
        pre[i] = samples[i] - config_.preemph * samples[i - 1];
      }
    }

    return ComputeFrameRange(pre, 0, 0, NumFrames(n_samples));
  }

  // Steps 2-7, for an arbitrary frame range over an arbitrary window of the
  // already-pre-emphasised signal.
  //
  // Absolute-index based rather than "give me the padded buffer" because the
  // streaming session needs to compute frames as audio arrives without ever
  // materialising the whole clip. Sample indices outside `window` read as ZERO,
  // which IS the padding convention (torch.stft center=True,
  // pad_mode="constant"), so neither caller builds a padded buffer.
  //
  // Reflect padding is the seductive wrong answer here. Measured against what
  // NeMo actually produced, reflect puts frame 0 of a clip several units out
  // in the log domain and the clip's final frame out as well; with zeros, the
  // whole clip agrees with NeMo. Reflect makes the first and last
  // frames see mirrored signal where NeMo sees silence — and it is also what
  // would make a streaming implementation of this function impossible without
  // buffering the entire clip.
  std::vector<float> ComputeFrameRange(const std::vector<float> &window,
                                       int64_t window_start,
                                       int64_t first_frame,
                                       int32_t n_frames) const {
    const int32_t n_fft = config_.n_fft;
    const int32_t n_bins = n_fft / 2 + 1;
    const int32_t n_mels = config_.n_mels;
    const int32_t hop = config_.hop_length;
    const int64_t window_end = window_start + static_cast<int64_t>(window.size());

    std::vector<float> out(static_cast<size_t>(std::max(n_frames, 0)) * n_mels,
                           0.0f);
    std::vector<float> buf(n_fft);
    std::vector<float> power(n_bins);

    for (int32_t t = 0; t != n_frames; ++t) {
      // Steps 3+4: frame and window. The frame is taken at full n_fft width
      // because the window is already padded to n_fft; taking a win_length
      // frame and padding it on the right instead would shift the signal by
      // (n_fft - win_length)/2 samples relative to torch.
      //
      // Frame `f` is centred on sample f*hop, so it spans signal samples
      // [f*hop - n_fft/2, f*hop + n_fft/2).
      const int64_t start = (first_frame + t) * hop - n_fft / 2;
      for (int32_t i = 0; i != n_fft; ++i) {
        const int64_t idx = start + i;
        const float s = (idx >= window_start && idx < window_end)
                            ? window[static_cast<size_t>(idx - window_start)]
                            : 0.0f;
        buf[i] = s * padded_window_[i];
      }

      rfft_.Compute(buf.data());

      // Step 5: power spectrum. knf::Rfft packs the result as
      // [R(0), R(n/2), R(1), I(1), R(2), I(2), ...]. Its imaginary sign is the
      // conjugate of numpy's; irrelevant for a power spectrum but not for
      // anything that consumes the phase.
      power[0] = buf[0] * buf[0];
      power[n_bins - 1] = buf[1] * buf[1];
      for (int32_t k = 1; k < n_bins - 1; ++k) {
        const float re = buf[2 * k];
        const float im = buf[2 * k + 1];
        power[k] = re * re + im * im;
      }

      // Steps 6+7: mel projection, then log with an ADDITIVE zero guard.
      float *dst = out.data() + static_cast<size_t>(t) * n_mels;
      for (int32_t m = 0; m != n_mels; ++m) {
        const float *w = mel_filters_.data() + static_cast<size_t>(m) * n_bins;
        float acc = 0.0f;
        for (int32_t k = 0; k != n_bins; ++k) {
          acc += w[k] * power[k];
        }
        dst[m] = std::log(acc + config_.log_zero_guard);
      }
    }

    return out;
  }

  const std::vector<float> &MelFilters() const { return mel_filters_; }

  int32_t NumMels() const { return config_.n_mels; }

 private:
  NemoFrontendConfig config_;
  // Mutable because knf::Rfft::Compute writes its twiddle scratch. Consequence:
  // Compute() is const but NOT thread-safe on one instance. The diarization
  // session and each NeMo-mode feature extractor own exactly one frontend and
  // run it from one thread at a time; sharing an instance across threads
  // needs one Rfft per thread, not a lock.
  mutable knf::Rfft rfft_;
  std::vector<float> mel_filters_;
  std::vector<float> padded_window_;
};

bool NemoFrontendConfig::Validate() const {
  if (n_fft < win_length) {
    SHERPA_ONNX_LOGE(
        "NeMo frontend: n_fft (%d) must be >= win_length (%d) — the "
        "window is zero-padded symmetrically into the FFT size",
        n_fft, win_length);
    return false;
  }

  if ((n_fft - win_length) % 2 != 0) {
    SHERPA_ONNX_LOGE(
        "NeMo frontend: n_fft - win_length (%d) must be even so the "
        "window pads symmetrically, as torch.stft does",
        n_fft - win_length);
    return false;
  }

  if (n_fft % 2 != 0) {
    SHERPA_ONNX_LOGE("NeMo frontend: n_fft (%d) must be even", n_fft);
    return false;
  }

  if (hop_length <= 0 || n_mels <= 0 || sample_rate <= 0) {
    SHERPA_ONNX_LOGE(
        "NeMo frontend: hop_length (%d), n_mels (%d) and sample_rate "
        "(%d) must all be positive",
        hop_length, n_mels, sample_rate);
    return false;
  }

  if (high_freq <= low_freq) {
    SHERPA_ONNX_LOGE(
        "NeMo frontend: high_freq (%f) must exceed low_freq (%f)",
        high_freq, low_freq);
    return false;
  }

  if (high_freq > sample_rate / 2.0f) {
    SHERPA_ONNX_LOGE(
        "NeMo frontend: high_freq (%f) exceeds Nyquist (%f)", high_freq,
        sample_rate / 2.0f);
    return false;
  }

  return true;
}

std::string NemoFrontendConfig::ToString() const {
  std::ostringstream os;
  os << "NemoFrontendConfig(";
  os << "sample_rate=" << sample_rate << ", ";
  os << "n_fft=" << n_fft << ", ";
  os << "win_length=" << win_length << ", ";
  os << "hop_length=" << hop_length << ", ";
  os << "n_mels=" << n_mels << ", ";
  os << "preemph=" << preemph << ", ";
  os << "log_zero_guard=" << log_zero_guard << ", ";
  os << "low_freq=" << low_freq << ", ";
  os << "high_freq=" << high_freq << ")";
  return os.str();
}

NemoFrontend::NemoFrontend(const NemoFrontendConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

NemoFrontend::~NemoFrontend() = default;

int32_t NemoFrontend::NumFrames(int32_t n_samples) const {
  return impl_->NumFrames(n_samples);
}

std::vector<float> NemoFrontend::Compute(
    const std::vector<float> &samples) const {
  return impl_->Compute(samples);
}

std::vector<float> NemoFrontend::ComputeFrameRange(
    const std::vector<float> &window, int64_t window_start, int64_t first_frame,
    int32_t n_frames) const {
  return impl_->ComputeFrameRange(window, window_start, first_frame, n_frames);
}

const std::vector<float> &NemoFrontend::MelFilters() const {
  return impl_->MelFilters();
}

int32_t NemoFrontend::NumMels() const { return impl_->NumMels(); }

NemoPreemphasisBuffer::NemoPreemphasisBuffer(const NemoFrontendConfig &config)
    : preemph_(config.preemph),
      hop_length_(config.hop_length),
      half_fft_(config.n_fft / 2) {}

void NemoPreemphasisBuffer::Accept(const float *samples, int32_t n) {
  if (n <= 0) {
    return;
  }

  const size_t old = samples_.size();
  samples_.resize(old + n);
  float prev = last_raw_sample_;
  for (int32_t i = 0; i != n; ++i) {
    const float x = samples[i];
    samples_[old + i] =
        (num_accepted_ == 0 && i == 0) ? x : x - preemph_ * prev;
    prev = x;
  }
  last_raw_sample_ = prev;
  num_accepted_ += n;
}

void NemoPreemphasisBuffer::DropBeforeFrame(int64_t next_frame) {
  const int64_t keep_from =
      std::max<int64_t>(0, next_frame * hop_length_ - half_fft_);
  if (keep_from <= start_) {
    return;
  }
  const int64_t drop = std::min<int64_t>(
      keep_from - start_, static_cast<int64_t>(samples_.size()));
  samples_.erase(samples_.begin(), samples_.begin() + drop);
  start_ += drop;
}

void NemoPreemphasisBuffer::Reset() {
  samples_.clear();
  start_ = 0;
  num_accepted_ = 0;
  last_raw_sample_ = 0;
}

}  // namespace sherpa_onnx
