// sherpa-onnx/csrc/streaming-sortformer-diarization.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: see streaming-sortformer-diarization.h for the feature, the three
// phases and why ingestion is compute.

#include "sherpa-onnx/csrc/streaming-sortformer-diarization.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/nemo-frontend.h"
#include "sherpa-onnx/csrc/sortformer-aosc.h"
#include "sherpa-onnx/csrc/sortformer-model.h"

namespace sherpa_onnx {

class StreamingSortformerDiarization::Impl {
 public:
  explicit Impl(const SortformerDiarizationConfig &config)
      : config_(config),
        frontend_(config.frontend),
        model_(std::make_unique<SortformerModel>(config)),
        aosc_(config.streaming),
        preemph_buffer_(config.frontend) {
    Reset();
  }

  template <typename Manager>
  Impl(Manager *mgr, const SortformerDiarizationConfig &config)
      : config_(config),
        frontend_(config.frontend),
        model_(std::make_unique<SortformerModel>(mgr, config)),
        aosc_(config.streaming),
        preemph_buffer_(config.frontend) {
    Reset();
  }

  void Reset() {
    aosc_.Reset();
    preemph_buffer_.Reset();
    n_features_ = 0;
    features_.clear();
    features_start_ = 0;
    chunk_start_frame_ = 0;
    track_.clear();
    segments_.clear();
    finalized_ = false;
  }

  void AcceptWaveform(const float *samples, int32_t n) {
    if (finalized_) {
      SHERPA_ONNX_LOGE(
          "sortformer: AcceptWaveform after Finalize. The session is done; "
          "create a new one rather than continuing a finished recording.");
      SHERPA_ONNX_EXIT(-1);
    }
    if (n <= 0) {
      return;
    }

    // Step 1 of the frontend, incrementally. Pre-emphasis is a one-sample
    // recurrence, so the previous call's LAST RAW sample has to survive into
    // this one; NemoPreemphasisBuffer carries it, as it does for the feature
    // extractor's NeMo mode.
    preemph_buffer_.Accept(samples, n);

    ComputeReadyFeatures(/*flush=*/false);
    RunReadyChunks(/*flush=*/false);
    TrimBuffers();
  }

  const std::vector<SortformerSegment> &Finalize() {
    if (finalized_) {
      return segments_;
    }
    finalized_ = true;

    ComputeReadyFeatures(/*flush=*/true);
    RunReadyChunks(/*flush=*/true);

    const int32_t n_spk = config_.streaming.n_spk;
    const int32_t n_frames = static_cast<int32_t>(track_.size() / n_spk);

    // NeMo trims the track to the frame count the audio actually covers; the
    // last chunk can emit encoder frames past the end of the signal.
    const int32_t n_out = static_cast<int32_t>(
        std::ceil(static_cast<double>(n_features_) /
                  config_.streaming.subsampling_factor));
    const int32_t n_valid = std::min(n_frames, n_out);
    track_.resize(static_cast<size_t>(n_valid) * n_spk);

    const float clip_duration =
        static_cast<float>(preemph_buffer_.NumAccepted()) /
        config_.frontend.sample_rate;
    segments_ = SortformerDisjointSegments(
        track_, n_valid, n_spk, config_.streaming.FrameDurationSeconds(),
        config_.post_processing, clip_duration);
    return segments_;
  }

  float ProcessedSeconds() const {
    const int32_t n_spk = config_.streaming.n_spk;
    return static_cast<float>(track_.size() / n_spk) *
           config_.streaming.FrameDurationSeconds();
  }

  float AcceptedSeconds() const {
    return static_cast<float>(preemph_buffer_.NumAccepted()) /
           config_.frontend.sample_rate;
  }

  const std::vector<float> &RawTrack() const { return track_; }

  int32_t NumFrames() const {
    return static_cast<int32_t>(track_.size() / config_.streaming.n_spk);
  }

  const std::string &ModelSha256() const { return model_->ModelSha256(); }

  const std::vector<int32_t> &WorkerThreadIds() const {
    return model_->WorkerThreadIds();
  }

 private:
  // Frame f spans samples [f*hop - n_fft/2, f*hop + n_fft/2), so it is fully
  // determined once n_samples reaches f*hop + n_fft/2. Beyond the end of the
  // signal the convention is zeros, which is why `flush` can simply compute the
  // remaining frames with no extra buffering.
  void ComputeReadyFeatures(bool flush) {
    const int32_t hop = config_.frontend.hop_length;
    const int32_t half = config_.frontend.n_fft / 2;

    const int64_t n_samples = preemph_buffer_.NumAccepted();

    int64_t ready = 0;
    if (flush) {
      ready = frontend_.NumFrames(static_cast<int32_t>(n_samples));
    } else {
      ready = (n_samples - half) / hop + 1;
      ready = std::max<int64_t>(ready, 0);
    }
    if (ready <= n_features_) {
      return;
    }

    const int32_t n_new = static_cast<int32_t>(ready - n_features_);
    const std::vector<float> block = frontend_.ComputeFrameRange(
        preemph_buffer_.Samples(), preemph_buffer_.Start(), n_features_,
        n_new);
    features_.insert(features_.end(), block.begin(), block.end());
    n_features_ = ready;
  }

  // Mirrors NeMo's streaming_feat_loader: chunk k covers feature frames
  // [start - left, end + right) with left/right context clamped at the clip
  // edges. A chunk may only run once its RIGHT context exists, or the encoder
  // sees a truncated receptive field at the chunk's edge — which does not fail,
  // it degrades.
  void RunReadyChunks(bool flush) {
    const auto &s = config_.streaming;
    const int32_t step = s.chunk_len * s.subsampling_factor;
    const int32_t n_mels = config_.frontend.n_mels;

    while (chunk_start_frame_ < n_features_) {
      const int64_t left =
          std::min<int64_t>(s.chunk_left_context * s.subsampling_factor,
                            chunk_start_frame_);
      const int64_t end = chunk_start_frame_ + step;
      const int64_t want_right = s.chunk_right_context * s.subsampling_factor;

      int64_t chunk_end = 0;
      int64_t right = 0;
      if (end + want_right <= n_features_) {
        chunk_end = end;
        right = want_right;
      } else if (flush) {
        chunk_end = std::min(end, n_features_);
        right = std::min(want_right, n_features_ - chunk_end);
      } else {
        return;  // wait for more audio
      }

      const int64_t from = chunk_start_frame_ - left;
      const int64_t to = chunk_end + right;
      const int32_t n_chunk_frames = static_cast<int32_t>(to - from);

      const float *feat = features_.data() +
                          static_cast<size_t>(from - features_start_) * n_mels;
      SortformerModelOutput out =
          model_->Forward(feat, n_chunk_frames, aosc_.Spkcache(),
                          aosc_.SpkcacheLengths(), aosc_.Fifo(),
                          aosc_.FifoLengths());

      const int32_t lc = static_cast<int32_t>(
          std::llround(static_cast<double>(left) / s.subsampling_factor));
      const int32_t rc = static_cast<int32_t>(
          std::ceil(static_cast<double>(right) / s.subsampling_factor));

      const std::vector<float> chunk_preds = aosc_.Step(
          out.chunk_pre_encode_embs.data(), out.n_chunk_enc,
          out.chunk_pre_encode_length, out.preds.data(), out.n_packed, lc, rc);
      track_.insert(track_.end(), chunk_preds.begin(), chunk_preds.end());

      chunk_start_frame_ = chunk_end;
      if (chunk_end == n_features_) {
        break;
      }
    }
  }

  // Drops audio and features no future chunk can read. Without this a long
  // recording accumulates its whole waveform AND its whole feature matrix
  // beside a 469 MB model.
  void TrimBuffers() {
    const int32_t n_mels = config_.frontend.n_mels;
    const auto &s = config_.streaming;

    const int64_t keep_frame_from =
        std::max<int64_t>(0, chunk_start_frame_ -
                                 s.chunk_left_context * s.subsampling_factor);
    if (keep_frame_from > features_start_) {
      const size_t drop =
          static_cast<size_t>(keep_frame_from - features_start_) * n_mels;
      features_.erase(features_.begin(), features_.begin() + drop);
      features_start_ = keep_frame_from;
    }

    // The samples no uncomputed frame reads, the same trim the feature
    // extractor's NeMo mode applies.
    preemph_buffer_.DropBeforeFrame(n_features_);
  }

  SortformerDiarizationConfig config_;
  NemoFrontend frontend_;
  std::unique_ptr<SortformerModel> model_;
  SortformerAosc aosc_;

  // Pre-emphasised audio at absolute sample positions.
  NemoPreemphasisBuffer preemph_buffer_;

  // Log-mel features, covering absolute frames
  // [features_start_, n_features_).
  std::vector<float> features_;
  int64_t features_start_ = 0;
  int64_t n_features_ = 0;

  int64_t chunk_start_frame_ = 0;

  std::vector<float> track_;
  std::vector<SortformerSegment> segments_;
  bool finalized_ = false;
};

StreamingSortformerDiarization::StreamingSortformerDiarization(
    const SortformerDiarizationConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
StreamingSortformerDiarization::StreamingSortformerDiarization(
    Manager *mgr, const SortformerDiarizationConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

StreamingSortformerDiarization::~StreamingSortformerDiarization() = default;

void StreamingSortformerDiarization::AcceptWaveform(const float *samples,
                                                    int32_t n) {
  impl_->AcceptWaveform(samples, n);
}

float StreamingSortformerDiarization::ProcessedSeconds() const {
  return impl_->ProcessedSeconds();
}

float StreamingSortformerDiarization::AcceptedSeconds() const {
  return impl_->AcceptedSeconds();
}

const std::vector<SortformerSegment> &
StreamingSortformerDiarization::Finalize() {
  return impl_->Finalize();
}

const std::vector<float> &StreamingSortformerDiarization::RawTrack() const {
  return impl_->RawTrack();
}

int32_t StreamingSortformerDiarization::NumFrames() const {
  return impl_->NumFrames();
}

void StreamingSortformerDiarization::Reset() { impl_->Reset(); }

const std::string &StreamingSortformerDiarization::ModelSha256() const {
  return impl_->ModelSha256();
}

const std::vector<int32_t> &StreamingSortformerDiarization::WorkerThreadIds()
    const {
  return impl_->WorkerThreadIds();
}

#if __ANDROID_API__ >= 9
template StreamingSortformerDiarization::StreamingSortformerDiarization(
    AAssetManager *mgr, const SortformerDiarizationConfig &config);
#endif

#if __OHOS__
template StreamingSortformerDiarization::StreamingSortformerDiarization(
    NativeResourceManager *mgr, const SortformerDiarizationConfig &config);
#endif

}  // namespace sherpa_onnx
