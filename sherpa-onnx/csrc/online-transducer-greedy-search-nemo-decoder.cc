// sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  Sangeet Sagar

#include "sherpa-onnx/csrc/online-transducer-greedy-search-nemo-decoder.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

static Ort::Value BuildDecoderInput(int32_t token, OrtAllocator *allocator) {
  std::array<int64_t, 2> shape{1, 1};

  Ort::Value decoder_input =
      Ort::Value::CreateTensor<int32_t>(allocator, shape.data(), shape.size());

  int32_t *p = decoder_input.GetTensorMutableData<int32_t>();

  p[0] = token;

  return decoder_input;
}

static std::vector<Ort::Value> BuildStateViews(
    std::vector<Ort::Value> *states) {
  std::vector<Ort::Value> ans;
  ans.reserve(states->size());
  for (auto &v : *states) {
    ans.push_back(View(&v));
  }

  return ans;
}

static void DecodeOne(const float *encoder_out, int32_t num_rows,
                      int32_t num_cols, OnlineTransducerNeMoModel *model,
                      float blank_penalty, OnlineStream *s) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  int32_t vocab_size = model->VocabSize();
  int32_t blank_id = vocab_size - 1;

  auto &r = s->GetResult();

  Ort::Value decoder_out{nullptr};

  auto decoder_input = BuildDecoderInput(
      r.tokens.empty() ? blank_id : r.tokens.back(), model->Allocator());

  std::vector<Ort::Value> &last_decoder_states = s->GetNeMoDecoderStates();

  std::vector<Ort::Value> tmp_decoder_states;
  tmp_decoder_states = BuildStateViews(&last_decoder_states);

  // decoder_output_pair.second returns the next decoder state
  std::pair<Ort::Value, std::vector<Ort::Value>> decoder_output_pair =
      model->RunDecoder(std::move(decoder_input),
                        std::move(tmp_decoder_states));

  std::array<int64_t, 3> encoder_shape{1, num_cols, 1};

  bool emitted = false;

  int32_t max_symbols_per_frame = 10;

  // Decoder states from BEFORE the last emitted token. The stream must save
  // these (not the post-token states): the next chunk rebuilds decoder_input
  // from r.tokens.back() and runs the decoder on it again, so saving the
  // post-token states would feed the last token into the prediction network
  // TWICE per chunk boundary. The accumulated state drift makes the joiner
  // increasingly blank-dominant over long streams (words vanish). Mirrors
  // online-transducer-greedy-search-nemo-parakeet-unified-decoder.cc.
  std::vector<Ort::Value> last_token_decoder_states;

  for (int32_t t = 0; t != num_rows; ++t) {
    Ort::Value cur_encoder_out = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(encoder_out) + t * num_cols, num_cols,
        encoder_shape.data(), encoder_shape.size());

    for (int32_t q = 0; q != max_symbols_per_frame; ++q) {
      Ort::Value logit = model->RunJoiner(View(&cur_encoder_out),
                                          View(&decoder_output_pair.first));

      float *p_logit = logit.GetTensorMutableData<float>();
      if (blank_penalty > 0) {
        p_logit[blank_id] -= blank_penalty;
      }

      int32_t y = MaxElementIndex(p_logit, vocab_size);

      if (y != blank_id) {
        emitted = true;
        r.tokens.push_back(y);
        r.timestamps.push_back(t + r.frame_offset);
        r.num_trailing_blanks = 0;

        decoder_input = BuildDecoderInput(y, model->Allocator());

        auto next_decoder_output_pair = model->RunDecoder(
            std::move(decoder_input),
            BuildStateViews(&decoder_output_pair.second));
        last_token_decoder_states = std::move(decoder_output_pair.second);
        decoder_output_pair = std::move(next_decoder_output_pair);
      } else {
        ++r.num_trailing_blanks;
        break;
      }
    }
  }

  if (emitted) {
    s->SetNeMoDecoderStates(std::move(last_token_decoder_states));
  }

  r.frame_offset += num_rows;
}

void OnlineTransducerGreedySearchNeMoDecoder::Decode(Ort::Value encoder_out,
                                                     OnlineStream **ss,
                                                     int32_t n) const {
  auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);  // bs = 1

  if (batch_size != n) {
    SHERPA_ONNX_LOGE("Size mismatch! encoder_out.size(0) %d, n: %d",
                     static_cast<int32_t>(shape[0]), n);
    SHERPA_ONNX_EXIT(-1);
  }

  int32_t dim1 = static_cast<int32_t>(shape[1]);  // T
  int32_t dim2 = static_cast<int32_t>(shape[2]);  // encoder_out_dim

  const float *p = encoder_out.GetTensorData<float>();

  for (int32_t i = 0; i != batch_size; ++i) {
    const float *this_p = p + dim1 * dim2 * i;

    DecodeOne(this_p, dim1, dim2, model_, blank_penalty_, ss[i]);
  }
}

}  // namespace sherpa_onnx
