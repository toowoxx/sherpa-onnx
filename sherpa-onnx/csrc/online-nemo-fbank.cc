// sherpa-onnx/csrc/online-nemo-fbank.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: see online-nemo-fbank.h for the feature, the framing and the memory
// bounds this file implements.

#include "sherpa-onnx/csrc/online-nemo-fbank.h"

#include <algorithm>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

OnlineNemoFbank::OnlineNemoFbank(const NemoFrontendConfig &config)
    : config_(config), frontend_(config), preemph_buffer_(config) {}

void OnlineNemoFbank::AcceptWaveform(float sampling_rate,
                                     const float *waveform, int32_t n) {
  if (n <= 0) {
    return;
  }

  if (input_finished_) {
    SHERPA_ONNX_LOGE("AcceptWaveform called after InputFinished()");
    SHERPA_ONNX_EXIT(-1);
  }

  if (static_cast<int32_t>(sampling_rate) != config_.sample_rate) {
    SHERPA_ONNX_LOGE("NeMo features expect %d Hz, given %d Hz",
                     config_.sample_rate, static_cast<int32_t>(sampling_rate));
    SHERPA_ONNX_EXIT(-1);
  }

  preemph_buffer_.Accept(waveform, n);
  ComputeReadyFrames();
}

void OnlineNemoFbank::InputFinished() {
  if (input_finished_) {
    return;
  }
  input_finished_ = true;
  ComputeReadyFrames();
}

int32_t OnlineNemoFbank::NumFramesReady() const {
  return static_cast<int32_t>(num_frames_);
}

bool OnlineNemoFbank::IsLastFrame(int32_t frame) const {
  return input_finished_ && frame == num_frames_ - 1;
}

const float *OnlineNemoFbank::GetFrame(int32_t frame) const {
  if (frame < frames_start_ || frame >= num_frames_) {
    SHERPA_ONNX_LOGE(
        "NeMo feature frame %d is not held: held frames are [%d, %d)", frame,
        static_cast<int32_t>(frames_start_), static_cast<int32_t>(num_frames_));
    SHERPA_ONNX_EXIT(-1);
  }
  return frames_.data() +
         static_cast<size_t>(frame - frames_start_) * config_.n_mels;
}

void OnlineNemoFbank::Pop(int32_t n) {
  const int64_t drop =
      std::min<int64_t>(std::max<int32_t>(n, 0), num_frames_ - frames_start_);
  if (drop == 0) {
    return;
  }
  frames_.erase(frames_.begin(),
                frames_.begin() + static_cast<size_t>(drop) * config_.n_mels);
  frames_start_ += drop;
}

int32_t OnlineNemoFbank::Dim() const { return config_.n_mels; }

int32_t OnlineNemoFbank::NumFramesHeld() const {
  return static_cast<int32_t>(num_frames_ - frames_start_);
}

int64_t OnlineNemoFbank::NumSamplesHeld() const {
  return static_cast<int64_t>(preemph_buffer_.Samples().size());
}

void OnlineNemoFbank::ComputeReadyFrames() {
  const int32_t hop = config_.hop_length;
  const int32_t half = config_.n_fft / 2;
  const int64_t num_samples = preemph_buffer_.NumAccepted();

  // While streaming, frame f needs samples up to f*hop + half - 1. The branch
  // on num_samples >= half keeps the division off negative numbers, where
  // C++ rounds towards zero and would report frame 0 ready too early.
  int64_t ready = 0;
  if (input_finished_) {
    ready = num_samples / hop;
  } else if (num_samples >= half) {
    ready = (num_samples - half) / hop + 1;
  }

  if (ready > num_frames_) {
    const std::vector<float> block = frontend_.ComputeFrameRange(
        preemph_buffer_.Samples(), preemph_buffer_.Start(), num_frames_,
        static_cast<int32_t>(ready - num_frames_));
    frames_.insert(frames_.end(), block.begin(), block.end());
    num_frames_ = ready;
  }

  preemph_buffer_.DropBeforeFrame(num_frames_);
}

}  // namespace sherpa_onnx
