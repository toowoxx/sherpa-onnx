// sherpa-onnx/csrc/sortformer-diarization-config.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: see sortformer-diarization-config.h. This file only carries the
// ParseOptions plumbing and the validation; every constant and the reasoning
// behind it lives in the header.

#include "sherpa-onnx/csrc/sortformer-diarization-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void SortformerStreamingConfig::Register(ParseOptions *po) {
  po->Register("sortformer-chunk-len", &chunk_len,
               "Encoder frames of new audio per streaming chunk. 124 is "
               "NVIDIA's published 10.0s-latency configuration.");
  po->Register("sortformer-fifo-len", &fifo_len,
               "Capacity in encoder frames of the FIFO holding frames not yet "
               "committed to the speaker cache.");
  po->Register("sortformer-spkcache-len", &spkcache_len,
               "Capacity in encoder frames of the arrival-order speaker cache.");
  po->Register("sortformer-spkcache-update-period", &spkcache_update_period,
               "Nominal number of frames moved out of the FIFO per update.");
}

bool SortformerStreamingConfig::Validate() const {
  if (chunk_len <= 0 || fifo_len <= 0 || spkcache_len <= 0) {
    SHERPA_ONNX_LOGE(
        "sortformer: chunk_len (%d), fifo_len (%d) and spkcache_len (%d) must "
        "all be positive",
        chunk_len, fifo_len, spkcache_len);
    return false;
  }

  if (n_spk <= 0) {
    SHERPA_ONNX_LOGE("sortformer: n_spk must be positive, got %d", n_spk);
    return false;
  }

  if (spkcache_len % n_spk != 0) {
    SHERPA_ONNX_LOGE(
        "sortformer: spkcache_len (%d) must be a multiple of n_spk (%d) — the "
        "cache reserves an equal share per speaker",
        spkcache_len, n_spk);
    return false;
  }

  if (SpkcacheLenPerSpk() <= 0) {
    SHERPA_ONNX_LOGE(
        "sortformer: spkcache_len/n_spk (%d) must exceed "
        "spkcache_sil_frames_per_spk (%d)",
        spkcache_len / n_spk, spkcache_sil_frames_per_spk);
    return false;
  }

  if (subsampling_factor <= 0) {
    SHERPA_ONNX_LOGE("sortformer: subsampling_factor must be positive, got %d",
                     subsampling_factor);
    return false;
  }

  // The compression sentinel must be unreachable as a real flat index, or a
  // valid frame would be silently replaced by the silence profile.
  int64_t max_flat_index =
      static_cast<int64_t>(n_spk) *
      (spkcache_len + spkcache_update_period + fifo_len + chunk_len +
       spkcache_sil_frames_per_spk);
  if (max_index <= max_flat_index) {
    SHERPA_ONNX_LOGE(
        "sortformer: max_index (%d) must exceed the largest reachable flat "
        "score index (%lld)",
        max_index, static_cast<long long>(max_flat_index));  // NOLINT
    return false;
  }

  return true;
}

std::string SortformerStreamingConfig::ToString() const {
  std::ostringstream os;
  os << "SortformerStreamingConfig(";
  os << "chunk_len=" << chunk_len << ", ";
  os << "chunk_left_context=" << chunk_left_context << ", ";
  os << "chunk_right_context=" << chunk_right_context << ", ";
  os << "fifo_len=" << fifo_len << ", ";
  os << "spkcache_len=" << spkcache_len << ", ";
  os << "spkcache_update_period=" << spkcache_update_period << ", ";
  os << "subsampling_factor=" << subsampling_factor << ", ";
  os << "n_spk=" << n_spk << ", ";
  os << "emb_dim=" << emb_dim << ", ";
  os << "feat_dim=" << feat_dim << ", ";
  os << "spkcache_sil_frames_per_spk=" << spkcache_sil_frames_per_spk << ", ";
  os << "pred_score_threshold=" << pred_score_threshold << ", ";
  os << "sil_threshold=" << sil_threshold << ", ";
  os << "strong_boost_rate=" << strong_boost_rate << ", ";
  os << "weak_boost_rate=" << weak_boost_rate << ", ";
  os << "min_pos_scores_rate=" << min_pos_scores_rate << ", ";
  os << "scores_boost_latest=" << scores_boost_latest << ")";
  return os.str();
}

void SortformerPostProcessingConfig::Register(ParseOptions *po) {
  po->Register("sortformer-onset", &onset,
               "Posterior above which a speaker turns on (hysteresis).");
  po->Register("sortformer-offset", &offset,
               "Posterior below which an active speaker turns off. Must not "
               "exceed onset.");
  po->Register("sortformer-pad-onset", &pad_onset,
               "Seconds each detected region is extended to the left.");
  po->Register("sortformer-pad-offset", &pad_offset,
               "Seconds each detected region is extended to the right.");
  po->Register("sortformer-min-duration-on", &min_duration_on,
               "Seconds; shorter segments are dropped.");
  po->Register("sortformer-min-duration-off", &min_duration_off,
               "Seconds; shorter gaps between two segments of the same "
               "speaker are filled.");
}

bool SortformerPostProcessingConfig::Validate() const {
  if (offset > onset) {
    SHERPA_ONNX_LOGE(
        "sortformer post-processing: offset (%f) must not exceed onset (%f) — "
        "hysteresis requires the release threshold to sit at or below the "
        "attack threshold",
        offset, onset);
    return false;
  }

  if (min_duration_on < 0 || min_duration_off < 0 || pad_onset < 0 ||
      pad_offset < 0) {
    SHERPA_ONNX_LOGE(
        "sortformer post-processing: padding and duration parameters must be "
        "non-negative (pad_onset=%f, pad_offset=%f, min_duration_on=%f, "
        "min_duration_off=%f)",
        pad_onset, pad_offset, min_duration_on, min_duration_off);
    return false;
  }

  return true;
}

std::string SortformerPostProcessingConfig::ToString() const {
  std::ostringstream os;
  os << "SortformerPostProcessingConfig(";
  os << "onset=" << onset << ", ";
  os << "offset=" << offset << ", ";
  os << "pad_onset=" << pad_onset << ", ";
  os << "pad_offset=" << pad_offset << ", ";
  os << "min_duration_on=" << min_duration_on << ", ";
  os << "min_duration_off=" << min_duration_off << ")";
  return os.str();
}

void SortformerDiarizationConfig::Register(ParseOptions *po) {
  po->Register("sortformer-model", &model,
               "Path to the exported Streaming Sortformer ONNX model.");
  po->Register("num-threads", &num_threads,
               "Number of threads for the ONNX Runtime session.");
  po->Register("debug", &debug, "True to print model information.");
  po->Register("provider", &provider,
               "Execution provider. Only cpu is supported for Sortformer.");

  streaming.Register(po);
  post_processing.Register(po);
}

bool SortformerDiarizationConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("sortformer: --sortformer-model is required");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("sortformer: model '%s' does not exist", model.c_str());
    return false;
  }

  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("sortformer: num_threads must be >= 1, got %d",
                     num_threads);
    return false;
  }

  return streaming.Validate() && frontend.Validate() &&
         post_processing.Validate();
}

std::string SortformerDiarizationConfig::ToString() const {
  std::ostringstream os;
  os << "SortformerDiarizationConfig(";
  os << "model=\"" << model << "\", ";
  os << "streaming=" << streaming.ToString() << ", ";
  os << "frontend=" << frontend.ToString() << ", ";
  os << "post_processing=" << post_processing.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";
  return os.str();
}

}  // namespace sherpa_onnx
