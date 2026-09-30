// sherpa-onnx/csrc/features-nemo-test.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: the host tests for the feature extractor's NeMo mode
// (FeatureExtractorConfig::is_nemo), which the streaming NeMo transducer
// recognizer uses so that Nemotron sees the features NeMo trained it on.
//
// The mode's own behaviour is checked against the shared NeMo frontend's
// whole-signal computation (NemoFrontend::Compute) on a generated signal, so
// these tests need no input and run on any checkout.
//
//  1. Streaming: does the mode produce the same frames however the audio is
//     cut into chunks? Chunks of 1, 160, 1600 and 4000 samples and an uneven
//     sequence must each yield exactly the first n // 160 frames of the
//     whole-signal computation, bit for bit, and exactly n // 160 frames in
//     total. After every chunk, NumFramesReady must equal the number of frames
//     whose samples up to f * 160 + 255 have arrived. Frames that were read
//     and popped must be gone, or a stream of several hours would keep every
//     frame.
//  2. Edges: does the mode keep NeMo's frame count on very short input, and
//     the extractor's resampling? Inputs of 100, 200 and 300 samples, around
//     one hop and the 256 samples the first frame needs while streaming, must
//     end at InputFinished with the first n // 160 frames of the whole-signal
//     computation. Input at 48 kHz, fed in 4800-sample chunks, must give bit
//     for bit the whole-signal frames of the same audio brought to 16 kHz by
//     the extractor's resampler class (LinearResample) with the extractor's
//     settings, in the same chunks and without a flush, as the extractor runs
//     it.
//  3. Sample scale: does the mode read samples in [-1, 1] whatever the
//     config's normalize_samples says? With the flag false it must still give
//     the whole-signal frames bit for bit, where the other modes scale every
//     sample by 32768.
//  4. Model check: does a model get correct features or a clear error? A
//     model whose metadata asks for no feature normalization, as Nemotron's
//     does, gets the NeMo mode with its own mel bin count. One that asks for
//     a normalization (per_feature, all_features) is refused with an error
//     that names it, since the mode applies none. The tests call
//     SetNemoFeatureMode, which the streaming NeMo transducer recognizer
//     calls at load, so they need no model package.
//  5. Pre-emphasis buffer: does NemoPreemphasisBuffer, which the mode and
//     the Streaming Sortformer engine share, carry the stream and trim it
//     correctly? The engine loads its ONNX model in its constructor, so the
//     test uses the buffer as the engine does: frames computed once their
//     samples have all arrived, DropBeforeFrame with the frame count after
//     every call, and the centred count 1 + n // 160 at the end. In several
//     chunkings that must equal NemoFrontend::Compute bit for bit. Calls with
//     no or a negative sample count must change nothing, a DropBeforeFrame
//     past the held samples must drop only those and keep Start at the next
//     sample's absolute index, and after Reset the buffer must behave as a new
//     stream, its first sample passing through unchanged.
//
// The generated signal
//
// A tone rising across the band, two steady tones, low noise from a fixed
// linear congruential generator and a stretch of digital silence, so energy
// reaches every mel bin at some point and the log floor is reached. The test
// builds it, so the whole-signal reference and the streamed frames read the
// same samples.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/nemo-frontend.h"
#include "sherpa-onnx/csrc/online-nemo-fbank.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

namespace {

constexpr int32_t kHop = 160;
constexpr int32_t kHalfFft = 256;
constexpr int32_t kMels = 128;

// 5 s and 37 samples at 16 kHz, so the signal does not end on a hop.
constexpr int32_t kSignalLength = 5 * 16000 + 37;

// `n` samples of the signal the SPEC describes, at `sample_rate`.
std::vector<float> GeneratedSignal(int32_t sample_rate, int32_t n) {
  constexpr double kPi = 3.14159265358979323846;
  const double duration = static_cast<double>(n) / sample_rate;
  const int32_t silence_begin = n / 10 * 4;
  const int32_t silence_end = n / 2;

  std::vector<float> out(n);
  uint32_t state = 12345;
  for (int32_t i = 0; i != n; ++i) {
    state = state * 1664525u + 1013904223u;
    if (i >= silence_begin && i < silence_end) {
      out[i] = 0;
      continue;
    }
    const double t = static_cast<double>(i) / sample_rate;
    // Rises linearly from 50 Hz to 7950 Hz over the signal.
    const double rising =
        std::sin(2 * kPi * (50 * t + 7900 / (2 * duration) * t * t));
    const double steady =
        std::sin(2 * kPi * 440 * t) + 0.5 * std::sin(2 * kPi * 3000 * t);
    const double noise = static_cast<double>(state >> 8) / (1 << 24) - 0.5;
    out[i] = static_cast<float>(0.3 * rising + 0.2 * steady + 0.05 * noise);
  }
  return out;
}

// The first `n_frames` frames of the shared frontend's whole-signal
// computation of `samples`.
std::vector<float> WholeSignalFrames(const std::vector<float> &samples,
                                     int32_t n_frames) {
  NemoFrontendConfig config;
  NemoFrontend frontend(config);
  const std::vector<float> whole = frontend.Compute(samples);
  const size_t n_values = static_cast<size_t>(n_frames) * kMels;
  EXPECT_GE(whole.size(), n_values);
  return std::vector<float>(whole.begin(),
                            whole.begin() + std::min(n_values, whole.size()));
}

// The index of the first value where `a` and `b` differ, or a.size() when
// they are equal. Sizes are checked separately.
size_t FirstDifference(const std::vector<float> &a,
                       const std::vector<float> &b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i != n; ++i) {
    if (a[i] != b[i]) {
      return i;
    }
  }
  return a.size();
}

// Frames f with f * 160 + 255 < n_samples, i.e. whose samples have all arrived.
int32_t ExpectedReady(int64_t n_samples) {
  if (n_samples < kHalfFft) {
    return 0;
  }
  return static_cast<int32_t>((n_samples - kHalfFft) / kHop + 1);
}

FeatureExtractorConfig NemoConfig() {
  FeatureExtractorConfig config;
  config.is_nemo = true;
  config.feature_dim = kMels;
  return config;
}

// Feeds `samples` in chunks of the sizes in `chunks`, cycling through them, and
// reads every frame as soon as it is ready, the way the recognizer does: each
// GetFrames call pops the frames before its first index.
std::vector<float> Stream(const std::vector<float> &samples,
                          const std::vector<int32_t> &chunks,
                          const std::string &what) {
  FeatureExtractor extractor(NemoConfig());
  std::vector<float> out;
  int32_t next = 0;
  int64_t fed = 0;
  size_t k = 0;
  const int64_t total = static_cast<int64_t>(samples.size());

  while (fed < total) {
    const int32_t n = static_cast<int32_t>(
        std::min<int64_t>(chunks[k++ % chunks.size()], total - fed));
    extractor.AcceptWaveform(16000, samples.data() + fed, n);
    fed += n;

    const int32_t ready = extractor.NumFramesReady();
    if (ready != ExpectedReady(fed)) {
      ADD_FAILURE() << what << ": after " << fed << " samples, "
                    << ready << " frames ready, want " << ExpectedReady(fed);
      return out;
    }

    if (ready > next) {
      std::vector<float> block = extractor.GetFrames(next, ready - next);
      out.insert(out.end(), block.begin(), block.end());
      next = ready;
    }
  }

  extractor.InputFinished();
  const int32_t ready = extractor.NumFramesReady();
  if (ready > next) {
    std::vector<float> block = extractor.GetFrames(next, ready - next);
    out.insert(out.end(), block.begin(), block.end());
  }
  EXPECT_TRUE(extractor.IsLastFrame(ready - 1)) << what;
  EXPECT_FALSE(extractor.IsLastFrame(ready - 2)) << what;
  return out;
}

// Feeds `samples` at `sampling_rate` in calls of at most `chunk` samples to an
// extractor built from `config`, calls InputFinished and returns every frame;
// `num_frames` receives their count.
std::vector<float> FeedAll(const FeatureExtractorConfig &config,
                           const std::vector<float> &samples,
                           int32_t sampling_rate, int32_t chunk,
                           int32_t *num_frames) {
  FeatureExtractor extractor(config);
  for (size_t fed = 0; fed < samples.size(); fed += chunk) {
    const int32_t n = static_cast<int32_t>(
        std::min<size_t>(chunk, samples.size() - fed));
    extractor.AcceptWaveform(sampling_rate, samples.data() + fed, n);
  }
  extractor.InputFinished();
  *num_frames = extractor.NumFramesReady();
  if (*num_frames == 0) {
    return {};
  }
  return extractor.GetFrames(0, *num_frames);
}

// Feeds `samples` to a NemoPreemphasisBuffer in chunks of the sizes in
// `chunks`, cycling through them, and uses it as the Streaming Sortformer
// engine does: after every call it computes the frames whose samples have all
// arrived from NumAccepted, then calls DropBeforeFrame with the frame count.
// At the end it computes the remaining centred frames, 1 + n // 160, as the
// engine's flush does. Returns every frame.
std::vector<float> FramesThroughBuffer(const std::vector<float> &samples,
                                       const std::vector<int32_t> &chunks,
                                       const std::string &what) {
  NemoFrontendConfig config;
  NemoFrontend frontend(config);
  NemoPreemphasisBuffer buffer(config);
  std::vector<float> out;
  int64_t n_frames = 0;

  auto compute_up_to = [&](int64_t ready) {
    if (ready > n_frames) {
      const std::vector<float> block = frontend.ComputeFrameRange(
          buffer.Samples(), buffer.Start(), n_frames,
          static_cast<int32_t>(ready - n_frames));
      out.insert(out.end(), block.begin(), block.end());
      n_frames = ready;
    }
    buffer.DropBeforeFrame(n_frames);
  };

  const int64_t total = static_cast<int64_t>(samples.size());
  int64_t fed = 0;
  size_t k = 0;
  while (fed < total) {
    const int32_t n = static_cast<int32_t>(
        std::min<int64_t>(chunks[k++ % chunks.size()], total - fed));
    buffer.Accept(samples.data() + fed, n);
    fed += n;
    compute_up_to(ExpectedReady(buffer.NumAccepted()));

    const int64_t held = static_cast<int64_t>(buffer.Samples().size());
    if (buffer.Start() + held != buffer.NumAccepted()) {
      ADD_FAILURE() << what << ": after " << fed << " samples, Start "
                    << buffer.Start() << " plus " << held
                    << " held samples is not NumAccepted "
                    << buffer.NumAccepted();
      return out;
    }
  }

  compute_up_to(
      frontend.NumFrames(static_cast<int32_t>(buffer.NumAccepted())));
  return out;
}

}  // namespace

TEST(FeaturesNemo, ChunkingYieldsTheWholeSignalFrames) {
  const std::vector<float> samples = GeneratedSignal(16000, kSignalLength);
  const int32_t n_valid = kSignalLength / kHop;
  const std::vector<float> expected = WholeSignalFrames(samples, n_valid);
  ASSERT_EQ(expected.size(), static_cast<size_t>(n_valid) * kMels);

  const std::map<std::string, std::vector<int32_t>> chunkings = {
      {"chunks of 1", {1}},
      {"chunks of 160", {160}},
      {"chunks of 1600", {1600}},
      {"chunks of 4000", {4000}},
      {"uneven chunks", {7, 333, 1, 4096, 159, 161, 2500, 17, 256, 255}},
  };

  for (const auto &c : chunkings) {
    const std::vector<float> actual = Stream(samples, c.second, c.first);
    ASSERT_EQ(actual.size(), expected.size())
        << c.first << ": " << actual.size() / kMels << " frames, want "
        << n_valid;
    const size_t first_diff = FirstDifference(actual, expected);
    EXPECT_EQ(first_diff, actual.size())
        << c.first << ": first differing value at frame "
        << first_diff / kMels << ", bin " << first_diff % kMels;
  }
}

TEST(FeaturesNemo, ReadFramesAndSamplesAreReleased) {
  const std::vector<float> samples = GeneratedSignal(16000, kSignalLength);

  NemoFrontendConfig config;
  OnlineNemoFbank fbank(config);

  constexpr int32_t kChunk = 1600;
  int32_t popped = 0;
  int32_t max_frames_held = 0;
  int64_t max_samples_held = 0;
  for (size_t fed = 0; fed < samples.size(); fed += kChunk) {
    const int32_t n = static_cast<int32_t>(
        std::min<size_t>(kChunk, samples.size() - fed));
    fbank.AcceptWaveform(16000, samples.data() + fed, n);
    max_frames_held = std::max(max_frames_held, fbank.NumFramesHeld());
    max_samples_held = std::max(max_samples_held, fbank.NumSamplesHeld());

    // Read every ready frame, then drop it, as FeatureExtractor::GetFrames
    // does on its next call.
    const int32_t ready = fbank.NumFramesReady();
    for (int32_t f = popped; f != ready; ++f) {
      ASSERT_NE(fbank.GetFrame(f), nullptr);
    }
    fbank.Pop(ready - popped);
    popped = ready;
    EXPECT_EQ(fbank.NumFramesHeld(), 0);
  }

  // A chunk adds at most 1600 / 160 + 1 frames, and the samples kept are only
  // the ones the next frame still reads, fewer than n_fft (512), however long
  // the stream runs.
  EXPECT_LE(max_frames_held, kChunk / kHop + 1);
  EXPECT_LT(max_samples_held, 2 * kHalfFft);
  EXPECT_GT(popped, 0);

  EXPECT_DEATH(fbank.GetFrame(popped - 1), "is not held");
}

TEST(FeaturesNemo, ShortInputsEndWithNemosFrameCount) {
  const std::vector<float> samples = GeneratedSignal(16000, kSignalLength);

  // 100 samples are less than a hop, 200 are a hop but less than the 256
  // samples the first frame needs while streaming, and 300 are more than both.
  for (int32_t n : {100, 200, 300}) {
    FeatureExtractor extractor(NemoConfig());
    extractor.AcceptWaveform(16000, samples.data(), n);
    EXPECT_EQ(extractor.NumFramesReady(), ExpectedReady(n))
        << n << " samples, before InputFinished";
    extractor.InputFinished();
    const int32_t n_frames = extractor.NumFramesReady();
    ASSERT_EQ(n_frames, n / kHop) << n << " samples";
    if (n_frames == 0) {
      continue;
    }

    const std::vector<float> actual = extractor.GetFrames(0, n_frames);
    const std::vector<float> expected = WholeSignalFrames(
        std::vector<float>(samples.begin(), samples.begin() + n), n_frames);
    ASSERT_EQ(actual.size(), expected.size()) << n << " samples";
    const size_t first_diff = FirstDifference(actual, expected);
    EXPECT_EQ(first_diff, actual.size())
        << n << " samples: first differing value at frame "
        << first_diff / kMels << ", bin " << first_diff % kMels;
  }
}

TEST(FeaturesNemo, InputAtAnotherRateIsResampledTo16kHz) {
  const std::vector<float> samples_48k =
      GeneratedSignal(48000, 3 * 48000 + 101);
  constexpr int32_t kChunk = 4800;

  // FeatureExtractor resamples with LinearResample, a low-pass cutoff at 0.99
  // of the lower rate's Nyquist frequency and 6 zeros (features.cc), one call
  // per AcceptWaveform and never a flush. The reference runs the same class
  // with the same settings over the same chunks.
  constexpr float min_freq = 16000;
  const float cutoff = 0.99 * 0.5 * min_freq;
  constexpr int32_t kZeros = 6;

  LinearResample down(48000, 16000, cutoff, kZeros);
  std::vector<float> resampled;
  for (size_t fed = 0; fed < samples_48k.size(); fed += kChunk) {
    const int32_t n = static_cast<int32_t>(
        std::min<size_t>(kChunk, samples_48k.size() - fed));
    std::vector<float> part;
    down.Resample(samples_48k.data() + fed, n, false, &part);
    resampled.insert(resampled.end(), part.begin(), part.end());
  }

  const int32_t n_16k = static_cast<int32_t>(resampled.size() / kHop);
  ASSERT_GT(n_16k, 0);
  const std::vector<float> expected = WholeSignalFrames(resampled, n_16k);

  int32_t n_48k = 0;
  const std::vector<float> from_48k =
      FeedAll(NemoConfig(), samples_48k, 48000, kChunk, &n_48k);

  ASSERT_EQ(n_48k, n_16k) << "frame count at 48 kHz differs from the count "
                          << "of the same audio resampled to 16 kHz";
  ASSERT_EQ(from_48k.size(), expected.size());
  const size_t first_diff = FirstDifference(from_48k, expected);
  EXPECT_EQ(first_diff, from_48k.size())
      << "first differing value at frame " << first_diff / kMels << ", bin "
      << first_diff % kMels;
}

TEST(FeaturesNemo, SamplesAreNotScaledWhenNormalizeSamplesIsFalse) {
  const std::vector<float> samples = GeneratedSignal(16000, kSignalLength);
  const int32_t n_valid = kSignalLength / kHop;

  // normalize_samples = false makes the other modes scale every sample by
  // 32768, which would raise every NeMo log-mel value by log(32768^2).
  FeatureExtractorConfig config = NemoConfig();
  config.normalize_samples = false;

  int32_t n_frames = 0;
  const std::vector<float> actual =
      FeedAll(config, samples, 16000, 1600, &n_frames);
  ASSERT_EQ(n_frames, n_valid);

  const std::vector<float> expected = WholeSignalFrames(samples, n_valid);
  ASSERT_EQ(actual.size(), expected.size());
  const size_t first_diff = FirstDifference(actual, expected);
  EXPECT_EQ(first_diff, actual.size())
      << "first differing value at frame " << first_diff / kMels << ", bin "
      << first_diff % kMels;
}

TEST(FeaturesNemo, ModelWithoutNormalizationGetsTheNemoMode) {
  // Nemotron's metadata asks for no normalization; the model reports it as
  // an empty string.
  FeatureExtractorConfig config;
  config.feature_dim = 80;
  SetNemoFeatureMode(kMels, "", &config);
  EXPECT_TRUE(config.is_nemo);
  EXPECT_EQ(config.feature_dim, kMels);

  FeatureExtractor extractor(config);
  EXPECT_EQ(extractor.FeatureDim(), kMels);
}

TEST(FeaturesNemo, ModelAskingForNormalizationIsRefused) {
  for (const char *normalize_type : {"per_feature", "all_features"}) {
    FeatureExtractorConfig config;
    EXPECT_DEATH(SetNemoFeatureMode(kMels, normalize_type, &config),
                 std::string("feature normalization '") + normalize_type +
                     "'")
        << normalize_type;
  }
}

TEST(FeaturesNemo, PreemphasisBufferIgnoresEmptyAndNegativeCounts) {
  const std::vector<float> samples = GeneratedSignal(16000, 1000);
  NemoFrontendConfig config;

  NemoPreemphasisBuffer whole(config);
  whole.Accept(samples.data(), 1000);

  // Before the first sample, so the stream's first sample must still pass
  // through afterwards.
  NemoPreemphasisBuffer buffer(config);
  buffer.Accept(samples.data(), 0);
  buffer.Accept(samples.data(), -5);
  EXPECT_EQ(buffer.NumAccepted(), 0);
  EXPECT_EQ(buffer.Start(), 0);
  EXPECT_TRUE(buffer.Samples().empty());

  // Between calls, so the carried raw sample must stay sample 399.
  buffer.Accept(samples.data(), 400);
  const std::vector<float> held = buffer.Samples();
  buffer.Accept(samples.data() + 400, 0);
  buffer.Accept(samples.data() + 400, -1);
  EXPECT_EQ(buffer.NumAccepted(), 400);
  EXPECT_EQ(buffer.Start(), 0);
  EXPECT_EQ(buffer.Samples(), held);

  buffer.Accept(samples.data() + 400, 600);
  EXPECT_EQ(buffer.Samples(), whole.Samples());
}

TEST(FeaturesNemo, PreemphasisBufferUsedAsTheEngineDoesMatchesCompute) {
  const std::vector<float> samples = GeneratedSignal(16000, kSignalLength);
  NemoFrontendConfig config;
  NemoFrontend frontend(config);
  const std::vector<float> expected = frontend.Compute(samples);
  const int32_t n_frames = frontend.NumFrames(kSignalLength);
  ASSERT_EQ(expected.size(), static_cast<size_t>(n_frames) * kMels);

  const std::map<std::string, std::vector<int32_t>> chunkings = {
      {"chunks of 1", {1}},
      {"chunks of 160", {160}},
      {"chunks of 1600", {1600}},
      {"uneven chunks", {7, 333, 1, 4096, 159, 161, 2500, 17, 256, 255}},
  };

  for (const auto &c : chunkings) {
    const std::vector<float> actual =
        FramesThroughBuffer(samples, c.second, c.first);
    ASSERT_EQ(actual.size(), expected.size())
        << c.first << ": " << actual.size() / kMels << " frames, want "
        << n_frames;
    const size_t first_diff = FirstDifference(actual, expected);
    EXPECT_EQ(first_diff, actual.size())
        << c.first << ": first differing value at frame "
        << first_diff / kMels << ", bin " << first_diff % kMels;
  }
}

TEST(FeaturesNemo, PreemphasisBufferDropsAtMostWhatItHolds) {
  const std::vector<float> samples = GeneratedSignal(16000, 2000);
  NemoFrontendConfig config;

  NemoPreemphasisBuffer whole(config);
  whole.Accept(samples.data(), 2000);

  NemoPreemphasisBuffer buffer(config);
  buffer.Accept(samples.data(), 1000);

  // Frame 100 reads from sample 100 * 160 - 256 = 15744, past the 1000 held.
  buffer.DropBeforeFrame(100);
  EXPECT_TRUE(buffer.Samples().empty());
  EXPECT_EQ(buffer.Start(), 1000);
  EXPECT_EQ(buffer.NumAccepted(), 1000);

  // The next samples land at their absolute positions, pre-emphasised against
  // sample 999.
  buffer.Accept(samples.data() + 1000, 1000);
  EXPECT_EQ(buffer.Start(), 1000);
  EXPECT_EQ(buffer.Samples(), std::vector<float>(whole.Samples().begin() + 1000,
                                                 whole.Samples().end()));

  // Frame 3 reads from sample 224, which is already gone: nothing is dropped.
  buffer.DropBeforeFrame(3);
  EXPECT_EQ(buffer.Start(), 1000);
  EXPECT_EQ(buffer.Samples().size(), 1000u);

  buffer.DropBeforeFrame(100);
  EXPECT_TRUE(buffer.Samples().empty());
  EXPECT_EQ(buffer.Start(), 2000);
  EXPECT_EQ(buffer.NumAccepted(), 2000);
}

TEST(FeaturesNemo, PreemphasisBufferAfterResetIsANewStream) {
  const std::vector<float> first = GeneratedSignal(16000, 3000);
  const std::vector<float> second = GeneratedSignal(16000, 1500);
  NemoFrontendConfig config;

  NemoPreemphasisBuffer buffer(config);
  buffer.Accept(first.data(), 3000);
  buffer.DropBeforeFrame(10);
  ASSERT_GT(buffer.Start(), 0);
  // A carried raw sample of zero would hide a missing first-sample
  // passthrough after Reset.
  ASSERT_NE(first.back(), 0.0f);

  buffer.Reset();
  EXPECT_EQ(buffer.NumAccepted(), 0);
  EXPECT_EQ(buffer.Start(), 0);
  EXPECT_TRUE(buffer.Samples().empty());

  NemoPreemphasisBuffer fresh(config);
  int32_t fed = 0;
  for (int32_t n : {1, 499, 1000}) {
    buffer.Accept(second.data() + fed, n);
    fresh.Accept(second.data() + fed, n);
    fed += n;
  }
  ASSERT_FALSE(buffer.Samples().empty());
  EXPECT_EQ(buffer.Samples()[0], second[0]);
  EXPECT_EQ(buffer.Samples(), fresh.Samples());
  EXPECT_EQ(buffer.Start(), fresh.Start());
  EXPECT_EQ(buffer.NumAccepted(), fresh.NumAccepted());
}

}  // namespace sherpa_onnx
