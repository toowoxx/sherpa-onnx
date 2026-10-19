// sherpa-onnx/csrc/online-transducer-nemo-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#include "sherpa-onnx/csrc/online-transducer-nemo-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-encoder-backend.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transpose.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

namespace {

constexpr int64_t kDefaultAutoPromptId = 101;

// The metadata of an encoder that the caller runs, answered by its backend.
// The overload below lets the macros that read the ONNX metadata read it too.
struct ExternalEncoderMetadata {
  const NemoEncoderExternalBackend *backend;
};

// An overload in this namespace hides the one for the ONNX metadata.
using sherpa_onnx::LookupCustomModelMetaData;

std::string LookupCustomModelMetaData(const ExternalEncoderMetadata &meta_data,
                                      const char *key,
                                      OrtAllocator * /*allocator*/) {
  return meta_data.backend->metadata(key);
}

std::string StripQuotes(std::string s) {
  s = Trim(s);
  if (s.size() >= 2 &&
      ((s.front() == '"' && s.back() == '"') ||
       (s.front() == '\'' && s.back() == '\''))) {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

std::string NormalizeLanguage(std::string s) {
  s = StripQuotes(std::move(s));
  s = Trim(s);
  if (s.size() >= 2 && s.front() == '<' && s.back() == '>') {
    s = s.substr(1, s.size() - 2);
  }

  std::replace(s.begin(), s.end(), '_', '-');
  ToLowerCase(&s);
  return s;
}

void AddLanguagePromptId(
    const std::string &language, int64_t prompt_id,
    std::unordered_map<std::string, int64_t> *language_prompt_ids,
    std::vector<std::pair<std::string, int64_t>> *ordered_prompt_ids) {
  auto normalized = NormalizeLanguage(language);
  if (normalized.empty()) {
    return;
  }

  if (language_prompt_ids->emplace(normalized, prompt_id).second) {
    ordered_prompt_ids->push_back({normalized, prompt_id});
  }
}

void AddBaseLanguageAliases(
    std::unordered_map<std::string, int64_t> *language_prompt_ids,
    const std::vector<std::pair<std::string, int64_t>> &ordered_prompt_ids) {
  for (const auto &p : ordered_prompt_ids) {
    auto pos = p.first.find('-');
    if (pos == std::string::npos || pos == 0) {
      continue;
    }

    auto base = p.first.substr(0, pos);
    language_prompt_ids->emplace(std::move(base), p.second);
  }
}

// The metadata value is a JSON object mapping language strings to integer
// prompt ids, e.g. {"auto": 101, "en-US": 0, "ja-JP": 1}.
void ParseLanguagePromptDictionary(
    const std::string &value,
    std::unordered_map<std::string, int64_t> *language_prompt_ids,
    std::vector<std::pair<std::string, int64_t>> *ordered_prompt_ids) {
  auto j = nlohmann::json::parse(value, nullptr, /*allow_exceptions*/ false);
  if (j.is_discarded() || !j.is_object()) {
    return;
  }

  for (const auto &item : j.items()) {
    if (!item.value().is_number_integer()) {
      continue;
    }

    int64_t prompt_id = item.value().get<int64_t>();
    if (prompt_id < 0) {
      continue;
    }

    AddLanguagePromptId(item.key(), prompt_id, language_prompt_ids,
                        ordered_prompt_ids);
  }
}

template <typename Metadata>
bool ReadPromptIdFromMetadata(const Metadata &meta_data,
                              OrtAllocator *allocator, const char *key,
                              int64_t *prompt_id) {
  auto value = LookupCustomModelMetaData(meta_data, key, allocator);
  if (value.empty()) {
    return false;
  }

  int64_t id = -1;
  if (!ConvertStringToInteger(Trim(value), &id) || id < 0) {
    return false;
  }

  *prompt_id = id;
  return true;
}

}  // namespace

class OnlineTransducerNeMoModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.encoder), sess_opts_);
    InitEncoder(nullptr, 0);

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.decoder), sess_opts_);
    InitDecoder(nullptr, 0);

    joiner_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.joiner), sess_opts_);
    InitJoiner(nullptr, 0);
  }

  Impl(const OnlineModelConfig &config, NemoEncoderExternalBackend backend)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        backend_(std::move(backend)) {
    InitExternalEncoder();

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.decoder), sess_opts_);
    InitDecoder(nullptr, 0);

    joiner_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.transducer.joiner), sess_opts_);
    InitJoiner(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    auto buf = ReadFile(mgr, config.transducer.joiner);
    InitJoiner(buf.data(), buf.size());
  }

  std::vector<Ort::Value> RunEncoder(
      Ort::Value features, std::vector<Ort::Value> states,
      const std::vector<int64_t> &language_prompt_ids) {
    if (backend_.run) {
      return RunExternalEncoder(std::move(features), std::move(states),
                                language_prompt_ids);
    }

    Ort::Value &cache_last_channel = states[0];
    Ort::Value &cache_last_time = states[1];
    Ort::Value &cache_last_channel_len = states[2];

    int32_t batch_size = features.GetTensorTypeAndShapeInfo().GetShape()[0];

    std::array<int64_t, 1> length_shape{batch_size};

    Ort::Value length = Ort::Value::CreateTensor<int64_t>(
        allocator_, length_shape.data(), length_shape.size());

    int64_t *p_length = length.GetTensorMutableData<int64_t>();

    std::fill(p_length, p_length + batch_size, ChunkSize());

    // (B, T, C) -> (B, C, T)
    features = Transpose12(allocator_, &features);

    std::vector<Ort::Value> inputs;
    inputs.reserve(is_multilingual_ ? 6 : 5);
    inputs.push_back(std::move(features));
    inputs.push_back(View(&length));
    inputs.push_back(std::move(cache_last_channel));
    inputs.push_back(std::move(cache_last_time));
    inputs.push_back(std::move(cache_last_channel_len));

    Ort::Value prompt_id_tensor{nullptr};
    if (is_multilingual_) {
      std::array<int64_t, 1> prompt_id_shape{batch_size};
      prompt_id_tensor = Ort::Value::CreateTensor<int64_t>(
          allocator_, prompt_id_shape.data(), prompt_id_shape.size());
      int64_t *p_prompt_id = prompt_id_tensor.GetTensorMutableData<int64_t>();
      if (static_cast<int32_t>(language_prompt_ids.size()) == batch_size) {
        std::copy(language_prompt_ids.begin(), language_prompt_ids.end(),
                  p_prompt_id);
      } else {
        std::fill(p_prompt_id, p_prompt_id + batch_size, default_prompt_id_);
      }
      inputs.push_back(std::move(prompt_id_tensor));
    }

    auto out = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());
    // out[0]: logit
    // out[1] logit_length
    // out[2:] states_next
    //
    // we need to remove out[1]

    std::vector<Ort::Value> ans;
    ans.reserve(out.size() - 1);

    for (int32_t i = 0; i != out.size(); ++i) {
      if (i == 1) {
        continue;
      }

      ans.push_back(std::move(out[i]));
    }

    return ans;
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> RunDecoder(
      Ort::Value targets, std::vector<Ort::Value> states) {
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto shape = targets.GetTensorTypeAndShapeInfo().GetShape();
    int32_t batch_size = static_cast<int32_t>(shape[0]);

    std::vector<int64_t> length_shape = {batch_size};
    std::vector<int32_t> length_value(batch_size, 1);

    Ort::Value targets_length = Ort::Value::CreateTensor<int32_t>(
        memory_info, length_value.data(), batch_size, length_shape.data(),
        length_shape.size());

    std::vector<Ort::Value> decoder_inputs;
    decoder_inputs.reserve(2 + states.size());

    decoder_inputs.push_back(std::move(targets));
    decoder_inputs.push_back(std::move(targets_length));

    for (auto &s : states) {
      decoder_inputs.push_back(std::move(s));
    }

    auto decoder_out = decoder_sess_->Run(
        {}, decoder_input_names_ptr_.data(), decoder_inputs.data(),
        decoder_inputs.size(), decoder_output_names_ptr_.data(),
        decoder_output_names_ptr_.size());

    std::vector<Ort::Value> states_next;
    states_next.reserve(states.size());

    // decoder_out[0]: decoder_output
    // decoder_out[1]: decoder_output_length (discarded)
    // decoder_out[2:] states_next

    for (int32_t i = 0; i != states.size(); ++i) {
      states_next.push_back(std::move(decoder_out[i + 2]));
    }

    // we discard decoder_out[1]
    return {std::move(decoder_out[0]), std::move(states_next)};
  }

  Ort::Value RunJoiner(Ort::Value encoder_out, Ort::Value decoder_out) {
    std::array<Ort::Value, 2> joiner_input = {std::move(encoder_out),
                                              std::move(decoder_out)};
    auto logit = joiner_sess_->Run({}, joiner_input_names_ptr_.data(),
                                   joiner_input.data(), joiner_input.size(),
                                   joiner_output_names_ptr_.data(),
                                   joiner_output_names_ptr_.size());

    return std::move(logit[0]);
  }

  std::vector<Ort::Value> GetDecoderInitStates() {
    std::vector<Ort::Value> ans;
    ans.reserve(2);
    ans.push_back(View(&lstm0_));
    ans.push_back(View(&lstm1_));

    return ans;
  }

  int32_t ChunkSize() const { return window_size_; }

  int32_t ChunkShift() const { return chunk_shift_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }

  int32_t FeatureDim() const { return feat_dim_; }

  bool IsMultilingual() const { return is_multilingual_; }

  int64_t GetLanguagePromptId(const std::string &language) const {
    if (!is_multilingual_ || language.empty()) {
      return default_prompt_id_;
    }

    auto it = language_prompt_ids_.find(language);
    if (it != language_prompt_ids_.end()) {
      return it->second;
    }

    auto normalized = NormalizeLanguage(language);
    if (normalized.empty() || normalized == "auto") {
      return default_prompt_id_;
    }

    it = language_prompt_ids_.find(normalized);
    if (it != language_prompt_ids_.end()) {
      return it->second;
    }

    SHERPA_ONNX_LOGE(
        "Unsupported language '%s' for multilingual NeMo transducer; using "
        "auto",
        language.c_str());
    return default_prompt_id_;
  }

  int32_t VocabSize() const { return vocab_size_; }

  OrtAllocator *Allocator() { return allocator_; }

  std::string FeatureNormalizationMethod() const { return normalize_type_; }

  // Return a vector containing 3 tensors
  // - cache_last_channel
  // - cache_last_time_
  // - cache_last_channel_len
  std::vector<Ort::Value> GetEncoderInitStates() {
    std::vector<Ort::Value> ans;
    ans.reserve(3);
    ans.push_back(View(&cache_last_channel_));
    ans.push_back(View(&cache_last_time_));
    ans.push_back(View(&cache_last_channel_len_));

    return ans;
  }

  std::vector<Ort::Value> StackStates(
      std::vector<std::vector<Ort::Value>> states) const {
    int32_t batch_size = static_cast<int32_t>(states.size());
    if (batch_size == 1) {
      return std::move(states[0]);
    }

    std::vector<Ort::Value> ans;

    auto allocator = const_cast<Impl *>(this)->allocator_;

    // stack cache_last_channel
    std::vector<const Ort::Value *> buf(batch_size);

    // there are 3 states to be stacked
    for (int32_t i = 0; i != 3; ++i) {
      buf.clear();
      buf.reserve(batch_size);

      for (int32_t b = 0; b != batch_size; ++b) {
        assert(states[b].size() == 3);
        buf.push_back(&states[b][i]);
      }

      Ort::Value c{nullptr};
      if (i == 2) {
        c = Cat<int64_t>(allocator, buf, 0);
      } else {
        c = Cat(allocator, buf, 0);
      }

      ans.push_back(std::move(c));
    }

    return ans;
  }

  std::vector<std::vector<Ort::Value>> UnStackStates(
      std::vector<Ort::Value> states) {
    assert(states.size() == 3);

    std::vector<std::vector<Ort::Value>> ans;

    auto shape = states[0].GetTensorTypeAndShapeInfo().GetShape();
    int32_t batch_size = shape[0];
    ans.resize(batch_size);

    if (batch_size == 1) {
      ans[0] = std::move(states);
      return ans;
    }

    for (int32_t i = 0; i != 3; ++i) {
      std::vector<Ort::Value> v;
      if (i == 2) {
        v = Unbind<int64_t>(allocator_, &states[i], 0);
      } else {
        v = Unbind(allocator_, &states[i], 0);
      }

      assert(v.size() == batch_size);

      for (int32_t b = 0; b != batch_size; ++b) {
        ans[b].push_back(std::move(v[b]));
      }
    }

    return ans;
  }

 private:
  // The encoder runs outside of ONNX Runtime, once per stream of the batch.
  // Returns what the ONNX encoder returns without its length output: the
  // encoded frames (B, encoded_dim, T') and the three next states.
  std::vector<Ort::Value> RunExternalEncoder(
      Ort::Value features, std::vector<Ort::Value> states,
      const std::vector<int64_t> &language_prompt_ids) {
    const auto feature_shape = features.GetTensorTypeAndShapeInfo().GetShape();
    const int64_t batch_size = feature_shape[0];
    const int64_t n_frames = feature_shape[1];
    const int64_t channel_count = cache_last_channel_size();
    const int64_t time_count = cache_last_time_size();

    const int64_t max_frames = n_frames / subsampling_factor_ + 2;
    const int64_t encoded_capacity = backend_.encoded_dim * max_frames;
    std::vector<float> encoded(batch_size * encoded_capacity);
    std::vector<int32_t> encoded_frames(batch_size);

    std::array<int64_t, 4> channel_shape{batch_size, cache_last_channel_dim1_,
                                         cache_last_channel_dim2_,
                                         cache_last_channel_dim3_};
    std::array<int64_t, 4> time_shape{batch_size, cache_last_time_dim1_,
                                      cache_last_time_dim2_,
                                      cache_last_time_dim3_};
    Ort::Value next_channel = Ort::Value::CreateTensor<float>(
        allocator_, channel_shape.data(), channel_shape.size());
    Ort::Value next_time = Ort::Value::CreateTensor<float>(
        allocator_, time_shape.data(), time_shape.size());
    std::array<int64_t, 1> len_shape{batch_size};
    Ort::Value next_len = Ort::Value::CreateTensor<int64_t>(
        allocator_, len_shape.data(), len_shape.size());

    const float *p_features = features.GetTensorData<float>();
    const float *p_channel = states[0].GetTensorData<float>();
    const float *p_time = states[1].GetTensorData<float>();
    const int64_t *p_len = states[2].GetTensorData<int64_t>();

    for (int64_t b = 0; b != batch_size; ++b) {
      NemoEncoderRequest request;
      request.features = p_features + b * n_frames * backend_.feature_dim;
      request.n_frames = static_cast<int32_t>(n_frames);
      request.feature_dim = backend_.feature_dim;
      request.cache_last_channel = p_channel + b * channel_count;
      request.cache_last_channel_count = channel_count;
      request.cache_last_time = p_time + b * time_count;
      request.cache_last_time_count = time_count;
      request.cache_last_channel_len = p_len[b];
      request.prompt_id =
          static_cast<int64_t>(language_prompt_ids.size()) == batch_size
              ? language_prompt_ids[b]
              : default_prompt_id_;

      NemoEncoderResult result;
      result.encoded = encoded.data() + b * encoded_capacity;
      result.encoded_capacity = encoded_capacity;
      result.cache_last_channel =
          next_channel.GetTensorMutableData<float>() + b * channel_count;
      result.cache_last_time =
          next_time.GetTensorMutableData<float>() + b * time_count;

      if (!backend_.run(request, &result)) {
        SHERPA_ONNX_LOGE("The external NeMo encoder failed");
        SHERPA_ONNX_EXIT(-1);
      }

      encoded_frames[b] = result.encoded_frames;
      next_len.GetTensorMutableData<int64_t>()[b] =
          result.cache_last_channel_len;
    }

    const int32_t frames = encoded_frames[0];
    std::array<int64_t, 3> out_shape{batch_size, backend_.encoded_dim, frames};
    Ort::Value out = Ort::Value::CreateTensor<float>(
        allocator_, out_shape.data(), out_shape.size());
    const int64_t per_sample = backend_.encoded_dim * frames;
    for (int64_t b = 0; b != batch_size; ++b) {
      std::copy(encoded.begin() + b * encoded_capacity,
                encoded.begin() + b * encoded_capacity + per_sample,
                out.GetTensorMutableData<float>() + b * per_sample);
    }

    std::vector<Ort::Value> ans;
    ans.reserve(4);
    ans.push_back(std::move(out));
    ans.push_back(std::move(next_channel));
    ans.push_back(std::move(next_time));
    ans.push_back(std::move(next_len));
    return ans;
  }

  int64_t cache_last_channel_size() const {
    return static_cast<int64_t>(cache_last_channel_dim1_) *
           cache_last_channel_dim2_ * cache_last_channel_dim3_;
  }

  int64_t cache_last_time_size() const {
    return static_cast<int64_t>(cache_last_time_dim1_) *
           cache_last_time_dim2_ * cache_last_time_dim3_;
  }

  void InitExternalEncoder() {
    if (!backend_.run || !backend_.metadata || backend_.feature_dim <= 0 ||
        backend_.encoded_dim <= 0) {
      SHERPA_ONNX_LOGE(
          "An external NeMo encoder needs run, metadata, feature_dim and "
          "encoded_dim");
      SHERPA_ONNX_EXIT(-1);
    }

    is_multilingual_ = backend_.multilingual;
    feat_dim_ = backend_.feature_dim;

    ExternalEncoderMetadata meta_data{&backend_};
    ReadEncoderMetadata(meta_data, nullptr);
    InitEncoderStates();
  }

  template <typename Metadata>
  void ReadEncoderMetadata(const Metadata &meta_data, OrtAllocator *allocator) {
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");

    // need to increase by 1 since the blank token is not included in computing
    // vocab_size in NeMo.
    vocab_size_ += 1;

    SHERPA_ONNX_READ_META_DATA(window_size_, "window_size");
    SHERPA_ONNX_READ_META_DATA(chunk_shift_, "chunk_shift");
    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");

    SHERPA_ONNX_READ_META_DATA_STR_ALLOW_EMPTY(normalize_type_,
                                               "normalize_type");
    SHERPA_ONNX_READ_META_DATA(pred_rnn_layers_, "pred_rnn_layers");
    SHERPA_ONNX_READ_META_DATA(pred_hidden_, "pred_hidden");

    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim1_,
                               "cache_last_channel_dim1");
    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim2_,
                               "cache_last_channel_dim2");
    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim3_,
                               "cache_last_channel_dim3");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim1_, "cache_last_time_dim1");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim2_, "cache_last_time_dim2");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim3_, "cache_last_time_dim3");

    if (normalize_type_ == "NA") {
      normalize_type_ = "";
    }

    if (is_multilingual_) {
      InitLanguagePromptIds(meta_data, allocator);
    }
  }

  void InitEncoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!encoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize encoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    is_multilingual_ =
        std::find(encoder_input_names_.begin(), encoder_input_names_.end(),
                  "prompt_index") != encoder_input_names_.end();

    feat_dim_ = encoder_sess_->GetInputTypeInfo(0)
                    .GetTensorTypeAndShapeInfo()
                    .GetShape()[1];

    // get meta data
    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---encoder---\n";
      PrintModelMetadata(os, meta_data);
      os << "feat_dim: " << feat_dim_ << "\n";
      os << "is_multilingual: " << (is_multilingual_ ? "1" : "0") << "\n";
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;
    ReadEncoderMetadata(meta_data, allocator);

    InitEncoderStates();
  }

  template <typename Metadata>
  void InitLanguagePromptIds(const Metadata &meta_data,
                             OrtAllocator *allocator) {
    auto dict = LookupCustomModelMetaData(meta_data, "prompt_dictionary",
                                          allocator);
    if (!dict.empty()) {
      ParseLanguagePromptDictionary(dict, &language_prompt_ids_,
                                    &ordered_prompt_ids_);
    }

    ReadPromptIdFromMetadata(meta_data, allocator, "auto_prompt_id",
                             &default_prompt_id_);

    auto it = language_prompt_ids_.find("auto");
    if (it != language_prompt_ids_.end()) {
      default_prompt_id_ = it->second;
    } else {
      AddLanguagePromptId("auto", default_prompt_id_, &language_prompt_ids_,
                          &ordered_prompt_ids_);
    }

    bool has_non_auto_prompt = false;
    for (const auto &p : ordered_prompt_ids_) {
      if (p.first != "auto") {
        has_non_auto_prompt = true;
        break;
      }
    }
    if (!has_non_auto_prompt) {
      SHERPA_ONNX_LOGE(
          "The encoder declares prompt_index, but usable prompt_dictionary "
          "metadata is missing; all languages will fall back to auto.");
    }

    AddBaseLanguageAliases(&language_prompt_ids_, ordered_prompt_ids_);
  }

  void InitEncoderStates() {
    std::array<int64_t, 4> cache_last_channel_shape{1,
                                                    cache_last_channel_dim1_,
                                                    cache_last_channel_dim2_,
                                                    cache_last_channel_dim3_};

    cache_last_channel_ = Ort::Value::CreateTensor<float>(
        allocator_, cache_last_channel_shape.data(),
        cache_last_channel_shape.size());

    Fill<float>(&cache_last_channel_, 0);

    std::array<int64_t, 4> cache_last_time_shape{
        1, cache_last_time_dim1_, cache_last_time_dim2_, cache_last_time_dim3_};

    cache_last_time_ = Ort::Value::CreateTensor<float>(
        allocator_, cache_last_time_shape.data(), cache_last_time_shape.size());

    Fill<float>(&cache_last_time_, 0);

    int64_t shape = 1;
    cache_last_channel_len_ =
        Ort::Value::CreateTensor<int64_t>(allocator_, &shape, 1);

    cache_last_channel_len_.GetTensorMutableData<int64_t>()[0] = 0;
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      decoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!decoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize decoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);

    InitDecoderStates();
  }

  void InitDecoderStates() {
    int32_t batch_size = 1;
    std::array<int64_t, 3> s0_shape{pred_rnn_layers_, batch_size, pred_hidden_};
    lstm0_ = Ort::Value::CreateTensor<float>(allocator_, s0_shape.data(),
                                             s0_shape.size());

    Fill<float>(&lstm0_, 0);

    std::array<int64_t, 3> s1_shape{pred_rnn_layers_, batch_size, pred_hidden_};

    lstm1_ = Ort::Value::CreateTensor<float>(allocator_, s1_shape.data(),
                                             s1_shape.size());

    Fill<float>(&lstm1_, 0);
  }

  void InitJoiner(void *model_data, size_t model_data_length) {
    if (model_data) {
      joiner_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!joiner_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize joiner session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(joiner_sess_.get(), &joiner_input_names_,
                  &joiner_input_names_ptr_);

    GetOutputNames(joiner_sess_.get(), &joiner_output_names_,
                   &joiner_output_names_ptr_);
  }

 private:
  OnlineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  // Set instead of encoder_sess_ when the caller runs the encoder.
  NemoEncoderExternalBackend backend_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;
  std::unique_ptr<Ort::Session> joiner_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<std::string> joiner_input_names_;
  std::vector<const char *> joiner_input_names_ptr_;

  std::vector<std::string> joiner_output_names_;
  std::vector<const char *> joiner_output_names_ptr_;

  int32_t window_size_ = 0;
  int32_t chunk_shift_ = 0;
  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 8;
  int32_t feat_dim_ = 80;
  bool is_multilingual_ = false;
  int64_t default_prompt_id_ = kDefaultAutoPromptId;
  std::unordered_map<std::string, int64_t> language_prompt_ids_;
  std::vector<std::pair<std::string, int64_t>> ordered_prompt_ids_;
  std::string normalize_type_;
  int32_t pred_rnn_layers_ = -1;
  int32_t pred_hidden_ = -1;

  // encoder states
  int32_t cache_last_channel_dim1_ = 0;
  int32_t cache_last_channel_dim2_ = 0;
  int32_t cache_last_channel_dim3_ = 0;
  int32_t cache_last_time_dim1_ = 0;
  int32_t cache_last_time_dim2_ = 0;
  int32_t cache_last_time_dim3_ = 0;

  // init encoder states
  Ort::Value cache_last_channel_{nullptr};
  Ort::Value cache_last_time_{nullptr};
  Ort::Value cache_last_channel_len_{nullptr};

  // init decoder states
  Ort::Value lstm0_{nullptr};
  Ort::Value lstm1_{nullptr};
};

OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    const OnlineModelConfig &config, NemoEncoderExternalBackend backend)
    : impl_(std::make_unique<Impl>(config, std::move(backend))) {}

template <typename Manager>
OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OnlineTransducerNeMoModel::~OnlineTransducerNeMoModel() = default;

std::vector<Ort::Value> OnlineTransducerNeMoModel::RunEncoder(
    Ort::Value features, std::vector<Ort::Value> states,
    const std::vector<int64_t> &language_prompt_ids) const {
  return impl_->RunEncoder(std::move(features), std::move(states),
                           language_prompt_ids);
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineTransducerNeMoModel::RunDecoder(Ort::Value targets,
                                      std::vector<Ort::Value> states) const {
  return impl_->RunDecoder(std::move(targets), std::move(states));
}

std::vector<Ort::Value> OnlineTransducerNeMoModel::GetDecoderInitStates()
    const {
  return impl_->GetDecoderInitStates();
}

Ort::Value OnlineTransducerNeMoModel::RunJoiner(Ort::Value encoder_out,
                                                Ort::Value decoder_out) const {
  return impl_->RunJoiner(std::move(encoder_out), std::move(decoder_out));
}

int32_t OnlineTransducerNeMoModel::ChunkSize() const {
  return impl_->ChunkSize();
}

int32_t OnlineTransducerNeMoModel::ChunkShift() const {
  return impl_->ChunkShift();
}

int32_t OnlineTransducerNeMoModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

int32_t OnlineTransducerNeMoModel::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OnlineTransducerNeMoModel::FeatureDim() const {
  return impl_->FeatureDim();
}

bool OnlineTransducerNeMoModel::IsMultilingual() const {
  return impl_->IsMultilingual();
}

int64_t OnlineTransducerNeMoModel::GetLanguagePromptId(
    const std::string &language) const {
  return impl_->GetLanguagePromptId(language);
}

OrtAllocator *OnlineTransducerNeMoModel::Allocator() const {
  return impl_->Allocator();
}

std::string OnlineTransducerNeMoModel::FeatureNormalizationMethod() const {
  return impl_->FeatureNormalizationMethod();
}

std::vector<Ort::Value> OnlineTransducerNeMoModel::GetEncoderInitStates()
    const {
  return impl_->GetEncoderInitStates();
}

std::vector<Ort::Value> OnlineTransducerNeMoModel::StackStates(
    std::vector<std::vector<Ort::Value>> states) const {
  return impl_->StackStates(std::move(states));
}

std::vector<std::vector<Ort::Value>> OnlineTransducerNeMoModel::UnStackStates(
    std::vector<Ort::Value> states) const {
  return impl_->UnStackStates(std::move(states));
}

#if __ANDROID_API__ >= 9
template OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
