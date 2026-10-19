// sherpa-onnx/csrc/online-nemo-encoder-backend-test.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// The host tests for the external encoder of the streaming NeMo transducer,
// the path where a platform runs the encoder (CoreML on Apple devices) and
// this library runs the feature extraction, the decoder, the joiner and the
// search around it.
//
// The tests go through the C API, because that is where a platform creates
// such a recognizer and where the ownership of the caller's user data is
// decided. They need no model: they check the rule that a create that fails
// returns NULL and never calls the caller's destroy, so the caller still owns
// its user data, as c-api.h promises. The success path needs a decoder and a
// joiner and is checked by running audio through a recognizer on a device or
// a Mac.
//
//  1. A backend without a run function or without a metadata function is
//     refused.
//  2. A config without a config is refused.
//  3. A decoder, a joiner or a tokens file that does not exist is refused.

#include "gtest/gtest.h"
#include "sherpa-onnx/c-api/c-api.h"

namespace sherpa_onnx {
namespace {

struct FakeEncoder {
  int32_t run_calls = 0;
  int32_t destroy_calls = 0;
};

int32_t FakeRun(void *user_data, const SherpaOnnxNemoEncoderRequest *,
                SherpaOnnxNemoEncoderResult *) {
  ++static_cast<FakeEncoder *>(user_data)->run_calls;
  return 0;
}

int32_t FakeMetadata(void *, const char *, char *, int32_t) { return 0; }

void FakeDestroy(void *user_data) {
  ++static_cast<FakeEncoder *>(user_data)->destroy_calls;
}

SherpaOnnxNemoEncoderExternalBackend BackendOf(FakeEncoder *encoder) {
  SherpaOnnxNemoEncoderExternalBackend backend = {};
  backend.user_data = encoder;
  backend.run = FakeRun;
  backend.metadata = FakeMetadata;
  backend.destroy = FakeDestroy;
  backend.feature_dim = 128;
  backend.encoded_dim = 1024;
  backend.multilingual = 1;
  return backend;
}

SherpaOnnxOnlineRecognizerConfig MissingFilesConfig() {
  SherpaOnnxOnlineRecognizerConfig config = {};
  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;
  config.model_config.transducer.decoder = "/nonexistent/decoder.onnx";
  config.model_config.transducer.joiner = "/nonexistent/joiner.onnx";
  config.model_config.tokens = "/nonexistent/tokens.txt";
  config.model_config.num_threads = 1;
  config.model_config.provider = "cpu";
  config.decoding_method = "greedy_search";
  return config;
}

TEST(OnlineNemoEncoderBackend, BackendWithoutRunOrMetadataIsRefused) {
  FakeEncoder encoder;
  auto config = MissingFilesConfig();

  auto backend = BackendOf(&encoder);
  backend.run = nullptr;
  EXPECT_EQ(SherpaOnnxCreateOnlineRecognizerWithNemoEncoderBackend(&config, &backend),
            nullptr);

  backend = BackendOf(&encoder);
  backend.metadata = nullptr;
  EXPECT_EQ(SherpaOnnxCreateOnlineRecognizerWithNemoEncoderBackend(&config, &backend),
            nullptr);

  EXPECT_EQ(encoder.run_calls, 0);
  EXPECT_EQ(encoder.destroy_calls, 0);
}

TEST(OnlineNemoEncoderBackend, MissingConfigOrBackendIsRefused) {
  FakeEncoder encoder;
  auto config = MissingFilesConfig();
  auto backend = BackendOf(&encoder);

  EXPECT_EQ(SherpaOnnxCreateOnlineRecognizerWithNemoEncoderBackend(nullptr, &backend),
            nullptr);
  EXPECT_EQ(SherpaOnnxCreateOnlineRecognizerWithNemoEncoderBackend(&config, nullptr),
            nullptr);
  EXPECT_EQ(encoder.destroy_calls, 0);
}

TEST(OnlineNemoEncoderBackend, MissingModelFilesAreRefusedWithoutDestroy) {
  FakeEncoder encoder;
  auto config = MissingFilesConfig();
  auto backend = BackendOf(&encoder);

  EXPECT_EQ(SherpaOnnxCreateOnlineRecognizerWithNemoEncoderBackend(&config, &backend),
            nullptr);

  EXPECT_EQ(encoder.run_calls, 0);
  EXPECT_EQ(encoder.destroy_calls, 0);
}

}  // namespace
}  // namespace sherpa_onnx
