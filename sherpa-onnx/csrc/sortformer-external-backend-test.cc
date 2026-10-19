// sherpa-onnx/csrc/sortformer-external-backend-test.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: the host tests for the external backend of the streaming Sortformer
// engine, the path where a platform runs the network (CoreML on Apple
// devices) and this library runs the frontend, the AOSC and the
// post-processing around it.
//
// The tests go through the C API, because that is where a platform creates
// such a session and where the ownership of the caller's user data is
// decided. A fake forward step stands in for the network, so the tests need
// no model and run on any checkout.
//
//  1. A session runs its chunks through the backend: audio of several chunks
//     calls the forward step once per chunk, Finalize returns the speaker the
//     fake reports, ModelId returns the backend's model_id, and destroying
//     the session calls the caller's destroy exactly once, never before.
//  2. A create that fails returns NULL and never calls destroy, so the caller
//     still owns its user data, as c-api.h promises. A config the engine
//     rejects and a backend without a forward step are both such failures.

#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/c-api/c-api.h"

namespace sherpa_onnx {
namespace {

constexpr int32_t kSlots = 4;
constexpr int32_t kEmbeddingDim = 512;
constexpr int32_t kSubsampling = 8;

// What the engine did with a fake network.
struct FakeNetwork {
  int32_t forward_calls = 0;
  int32_t destroy_calls = 0;
};

// Answers every chunk with the first slot alone, and sets the counts the
// engine reads the way the exported graph does: one packed row for each
// valid row of the cache and the FIFO and for each encoder frame of the
// chunk.
int32_t FakeForward(void *user_data,
                    const SherpaOnnxSortformerForwardRequest *request,
                    SherpaOnnxSortformerForwardResult *result) {
  auto *network = static_cast<FakeNetwork *>(user_data);
  ++network->forward_calls;

  const int32_t chunk_frames =
      (request->n_feat_frames + kSubsampling - 1) / kSubsampling;
  const int32_t packed =
      request->spkcache_length + request->fifo_length + chunk_frames;
  if (packed > result->preds_capacity ||
      chunk_frames > result->embs_capacity) {
    return 0;
  }

  for (int32_t row = 0; row != packed; ++row) {
    float *slots = result->preds + row * kSlots;
    std::fill(slots, slots + kSlots, 0.0f);
    slots[0] = 0.9f;
  }
  std::fill(result->chunk_pre_encode_embs,
            result->chunk_pre_encode_embs + chunk_frames * kEmbeddingDim,
            0.1f);
  result->n_packed = packed;
  result->n_chunk_enc = chunk_frames;
  result->chunk_pre_encode_length = chunk_frames;
  return 1;
}

void FakeDestroy(void *user_data) {
  ++static_cast<FakeNetwork *>(user_data)->destroy_calls;
}

SherpaOnnxSortformerExternalBackend BackendOf(FakeNetwork *network) {
  SherpaOnnxSortformerExternalBackend backend = {};
  backend.user_data = network;
  backend.forward = FakeForward;
  backend.destroy = FakeDestroy;
  backend.model_id = "fake-network";
  return backend;
}

// One second of a quiet tone at `sample_rate`.
std::vector<float> OneSecondOfTone(int32_t sample_rate) {
  std::vector<float> samples(sample_rate);
  for (int32_t i = 0; i != sample_rate; ++i) {
    samples[i] = 0.05f * std::sin(2.0f * 3.14159265f * 220.0f * i /
                                  static_cast<float>(sample_rate));
  }
  return samples;
}

TEST(SortformerExternalBackend, RunsTheChunksAndDestroysTheBackendOnce) {
  FakeNetwork network;
  // Zero keeps every default, the 10 s latency configuration.
  SherpaOnnxStreamingSortformerDiarizationConfig config = {};
  const SherpaOnnxSortformerExternalBackend backend = BackendOf(&network);

  const SherpaOnnxStreamingSortformerDiarization *sd =
      SherpaOnnxCreateStreamingSortformerDiarizationWithBackend(&config,
                                                                &backend);
  ASSERT_NE(sd, nullptr);
  EXPECT_STREQ(SherpaOnnxStreamingSortformerDiarizationModelId(sd),
               "fake-network");

  // 35 s: three full chunks of 9.92 s with their right context while the
  // audio arrives, and the tail at Finalize.
  const int32_t sample_rate =
      SherpaOnnxStreamingSortformerDiarizationGetSampleRate(sd);
  const std::vector<float> second = OneSecondOfTone(sample_rate);
  for (int32_t i = 0; i != 35; ++i) {
    ASSERT_EQ(SherpaOnnxStreamingSortformerDiarizationAcceptWaveform(
                  sd, second.data(), sample_rate),
              1);
  }
  EXPECT_EQ(network.forward_calls, 3);
  EXPECT_GT(SherpaOnnxStreamingSortformerDiarizationProcessedSeconds(sd), 0.0f);

  int32_t num_segments = 0;
  const SherpaOnnxStreamingSortformerDiarizationSegment *segments =
      SherpaOnnxStreamingSortformerDiarizationFinalize(sd, &num_segments);
  ASSERT_NE(segments, nullptr);
  EXPECT_EQ(network.forward_calls, 4);
  ASSERT_GE(num_segments, 1);
  for (int32_t i = 0; i != num_segments; ++i) {
    EXPECT_EQ(segments[i].speaker, 0);
  }
  SherpaOnnxStreamingSortformerDiarizationDestroySegments(segments);

  EXPECT_EQ(network.destroy_calls, 0);
  SherpaOnnxDestroyStreamingSortformerDiarization(sd);
  EXPECT_EQ(network.destroy_calls, 1);
}

TEST(SortformerExternalBackend, AFailedCreateNeverDestroysTheBackend) {
  FakeNetwork network;
  const SherpaOnnxSortformerExternalBackend backend = BackendOf(&network);

  // A chunk this long makes the flat score indices of the cache reach the
  // sentinel, which the engine's validation rejects.
  SherpaOnnxStreamingSortformerDiarizationConfig rejected = {};
  rejected.chunk_len = 1000000;
  EXPECT_EQ(SherpaOnnxCreateStreamingSortformerDiarizationWithBackend(
                &rejected, &backend),
            nullptr);

  SherpaOnnxStreamingSortformerDiarizationConfig config = {};
  SherpaOnnxSortformerExternalBackend no_forward = backend;
  no_forward.forward = nullptr;
  EXPECT_EQ(SherpaOnnxCreateStreamingSortformerDiarizationWithBackend(
                &config, &no_forward),
            nullptr);

  EXPECT_EQ(network.forward_calls, 0);
  EXPECT_EQ(network.destroy_calls, 0);
}

}  // namespace
}  // namespace sherpa_onnx
