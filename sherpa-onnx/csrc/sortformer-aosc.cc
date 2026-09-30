// sherpa-onnx/csrc/sortformer-aosc.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: see sortformer-aosc.h for the feature, the provenance of this
// algorithm, and the list of refactors that break it silently. This file is a
// line-by-line port of NeMo's SortformerModules; the section
// comments below name the NeMo function each block corresponds to so the two
// can be diffed by eye.

#include "sherpa-onnx/csrc/sortformer-aosc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();

// Indices of the k largest entries of `values`, in arbitrary order. Mirrors
// torch.topk(..., sorted=False) / np.argpartition. Ties are broken arbitrarily
// in every implementation involved; every tie reachable here is between
// infinities, whose destination (a disabled slot, or a silence pad) does not
// depend on which index won.
std::vector<int32_t> TopKIndices(const std::vector<float> &values, int32_t k) {
  const int32_t n = static_cast<int32_t>(values.size());
  k = std::min(k, n);
  if (k <= 0) {
    return {};
  }

  std::vector<int32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::nth_element(idx.begin(), idx.begin() + (n - k), idx.end(),
                   [&values](int32_t a, int32_t b) {
                     return values[a] < values[b];
                   });
  return std::vector<int32_t>(idx.begin() + (n - k), idx.end());
}

}  // namespace

SortformerAosc::SortformerAosc(const SortformerStreamingConfig &config)
    : config_(config) {
  Reset();
}

void SortformerAosc::Reset() {
  const int32_t s = config_.spkcache_len;
  const int32_t f = config_.fifo_len;
  const int32_t d = config_.emb_dim;
  const int32_t n = config_.n_spk;

  spkcache_.assign(static_cast<size_t>(s) * d, 0.0f);
  spkcache_preds_.assign(static_cast<size_t>(s) * n, 0.0f);
  spkcache_lengths_ = 0;
  spkcache_compressed_ = false;

  fifo_.assign(static_cast<size_t>(f) * d, 0.0f);
  fifo_preds_.assign(static_cast<size_t>(f) * n, 0.0f);
  fifo_lengths_ = 0;

  mean_sil_emb_.assign(d, 0.0f);
  n_sil_frames_ = 0;

  events_.clear();
}

// NeMo _get_log_pred_scores.
std::vector<float> SortformerAosc::LogPredScores(const float *preds,
                                                 int32_t n_frames) const {
  const int32_t n = config_.n_spk;
  const float thr = config_.pred_score_threshold;
  const float log_half = std::log(0.5f);

  std::vector<float> scores(static_cast<size_t>(n_frames) * n, 0.0f);
  for (int32_t t = 0; t != n_frames; ++t) {
    const float *p = preds + static_cast<size_t>(t) * n;
    float *dst = scores.data() + static_cast<size_t>(t) * n;

    // log(1 - p) summed over speakers: the frame's "everyone else is silent"
    // evidence, added to every speaker's own score. Computed first because the
    // per-speaker term subtracts its own contribution back out.
    float sum_log_1p = 0.0f;
    for (int32_t s = 0; s != n; ++s) {
      sum_log_1p += std::log(std::max(1.0f - p[s], thr));
    }

    for (int32_t s = 0; s != n; ++s) {
      const float log_p = std::log(std::max(p[s], thr));
      const float log_1p = std::log(std::max(1.0f - p[s], thr));
      dst[s] = log_p - log_1p + sum_log_1p - log_half;
    }
  }
  return scores;
}

// NeMo _disable_low_scores.
void SortformerAosc::DisableLowScores(const float *preds, int32_t n_frames,
                                      std::vector<float> *scores) const {
  const int32_t n = config_.n_spk;
  float *sc = scores->data();

  for (int32_t t = 0; t != n_frames; ++t) {
    const float *p = preds + static_cast<size_t>(t) * n;
    float *dst = sc + static_cast<size_t>(t) * n;
    for (int32_t s = 0; s != n; ++s) {
      if (!(p[s] > 0.5f)) {
        dst[s] = kNegInf;
      }
    }
  }

  // Per speaker: how many frames are confidently solo. Only speakers that
  // already have enough of those may discard their overlapped frames — a
  // speaker short on solo evidence keeps whatever it has.
  std::vector<int32_t> n_pos(n, 0);
  for (int32_t t = 0; t != n_frames; ++t) {
    const float *src = sc + static_cast<size_t>(t) * n;
    for (int32_t s = 0; s != n; ++s) {
      if (src[s] > 0.0f) {
        ++n_pos[s];
      }
    }
  }

  const int32_t min_pos = config_.MinPosScoresPerSpk();
  for (int32_t t = 0; t != n_frames; ++t) {
    const float *p = preds + static_cast<size_t>(t) * n;
    float *dst = sc + static_cast<size_t>(t) * n;
    for (int32_t s = 0; s != n; ++s) {
      const bool is_pos = dst[s] > 0.0f;
      const bool is_speech = p[s] > 0.5f;
      if (!is_pos && is_speech && n_pos[s] >= min_pos) {
        dst[s] = kNegInf;
      }
    }
  }
}

// NeMo _boost_topk_scores. In place, and the weak boost reads the strong
// boost's output — see the header.
void SortformerAosc::BoostTopK(int32_t n_frames, int32_t n_per_spk,
                               float scale_factor,
                               std::vector<float> *scores) const {
  if (n_per_spk <= 0) {
    return;
  }

  const int32_t n = config_.n_spk;
  const float bump = -scale_factor * std::log(0.5f);
  float *sc = scores->data();

  std::vector<float> column(n_frames);
  for (int32_t s = 0; s != n; ++s) {
    for (int32_t t = 0; t != n_frames; ++t) {
      column[t] = sc[static_cast<size_t>(t) * n + s];
    }
    for (int32_t t : TopKIndices(column, n_per_spk)) {
      sc[static_cast<size_t>(t) * n + s] += bump;
    }
  }
}

// NeMo _compress_spkcache.
void SortformerAosc::CompressSpkcache(const std::vector<float> &embs,
                                      const std::vector<float> &preds,
                                      int32_t n_frames) {
  const int32_t n = config_.n_spk;
  const int32_t d = config_.emb_dim;
  const int32_t s_cap = config_.spkcache_len;
  const int32_t n_sil = config_.spkcache_sil_frames_per_spk;

  std::vector<float> scores = LogPredScores(preds.data(), n_frames);
  DisableLowScores(preds.data(), n_frames, &scores);

  // Recency prior on the APPEND REGION, in physical coordinates. The candidate
  // buffer is [old cache | newly popped | zero padding]; everything from
  // spkcache_len on is "new this update".
  if (config_.scores_boost_latest > 0) {
    for (int32_t t = s_cap; t < n_frames; ++t) {
      float *dst = scores.data() + static_cast<size_t>(t) * n;
      for (int32_t sp = 0; sp != n; ++sp) {
        dst[sp] += config_.scores_boost_latest;
      }
    }
  }

  BoostTopK(n_frames, config_.StrongBoostPerSpk(), 2.0f, &scores);
  BoostTopK(n_frames, config_.WeakBoostPerSpk(), 1.0f, &scores);

  // Append n_sil rows of +inf per speaker. They always win a slot, which is how
  // each speaker's share of the cache reserves room for the silence profile.
  const int32_t padded_frames = n_frames + n_sil;
  const int32_t n_frames_no_sil = n_frames;

  // Speaker-major flatten: flat[sp * padded_frames + t]. Sorting the CHOSEN
  // flat indices is what orders the surviving cache by speaker and then by
  // arrival — the "arrival order" the cache is named for.
  std::vector<float> flat(static_cast<size_t>(n) * padded_frames);
  for (int32_t sp = 0; sp != n; ++sp) {
    for (int32_t t = 0; t != n_frames; ++t) {
      flat[static_cast<size_t>(sp) * padded_frames + t] =
          scores[static_cast<size_t>(t) * n + sp];
    }
    for (int32_t t = n_frames; t != padded_frames; ++t) {
      flat[static_cast<size_t>(sp) * padded_frames + t] = kPosInf;
    }
  }

  std::vector<int32_t> sel = TopKIndices(flat, s_cap);
  for (int32_t &v : sel) {
    if (flat[v] == kNegInf) {
      v = config_.max_index;
    }
  }
  std::sort(sel.begin(), sel.end());

  std::vector<float> out_embs(static_cast<size_t>(s_cap) * d, 0.0f);
  std::vector<float> out_preds(static_cast<size_t>(s_cap) * n, 0.0f);

  const int32_t n_sel = static_cast<int32_t>(sel.size());
  for (int32_t i = 0; i != s_cap; ++i) {
    bool disabled = i >= n_sel || sel[i] == config_.max_index;
    int32_t frame_idx = 0;
    if (!disabled) {
      frame_idx = sel[i] % padded_frames;
      // A selected silence pad has no frame behind it; it becomes a silence
      // slot too.
      if (frame_idx >= n_frames_no_sil) {
        disabled = true;
        frame_idx = 0;
      }
    }

    float *dst_emb = out_embs.data() + static_cast<size_t>(i) * d;
    float *dst_pred = out_preds.data() + static_cast<size_t>(i) * n;
    if (disabled) {
      std::copy(mean_sil_emb_.begin(), mean_sil_emb_.end(), dst_emb);
      // preds stay zero
    } else {
      const float *src_emb = embs.data() + static_cast<size_t>(frame_idx) * d;
      const float *src_pred = preds.data() + static_cast<size_t>(frame_idx) * n;
      std::copy(src_emb, src_emb + d, dst_emb);
      std::copy(src_pred, src_pred + n, dst_pred);
    }
  }

  spkcache_ = std::move(out_embs);
  spkcache_preds_ = std::move(out_preds);
}

std::vector<float> SortformerAosc::Step(const float *chunk_embs,
                                        int32_t t_chunk_enc,
                                        int32_t chunk_pre_encode_length,
                                        const float *preds, int32_t t_packed,
                                        int32_t lc, int32_t rc) {
  const int32_t s_cap = config_.spkcache_len;
  const int32_t f_cap = config_.fifo_len;
  const int32_t n = config_.n_spk;
  const int32_t d = config_.emb_dim;

  const int32_t max_chunk_len = t_chunk_enc - lc - rc;
  const int32_t max_pop_out_len =
      std::min(std::max({config_.spkcache_update_period, f_cap, max_chunk_len}),
               max_chunk_len + f_cap);
  const int32_t chunk_lengths =
      std::max(0, std::min(chunk_pre_encode_length - lc, max_chunk_len));

  const int32_t sl = spkcache_lengths_;
  const int32_t fl = fifo_lengths_;

  if (sl + fl + lc + chunk_lengths > t_packed) {
    SHERPA_ONNX_LOGE(
        "sortformer AOSC: packed predictions hold %d frames but the state "
        "needs %d (spkcache %d + fifo %d + lc %d + chunk %d). The graph's "
        "concat_and_pad layout and the tracked lengths have diverged.",
        t_packed, sl + fl + lc + chunk_lengths, sl, fl, lc, chunk_lengths);
    SHERPA_ONNX_EXIT(-1);
  }

  // NeMo _gather_async_predictions: split the packed predictions back onto the
  // three regions they were concatenated from.
  std::vector<float> current_spkcache_preds(static_cast<size_t>(s_cap) * n,
                                            0.0f);
  std::copy(preds, preds + static_cast<size_t>(sl) * n,
            current_spkcache_preds.begin());

  std::vector<float> current_fifo_preds(static_cast<size_t>(f_cap) * n, 0.0f);
  std::copy(preds + static_cast<size_t>(sl) * n,
            preds + static_cast<size_t>(sl + fl) * n,
            current_fifo_preds.begin());

  std::vector<float> chunk_preds(static_cast<size_t>(std::max(max_chunk_len, 0)) * n,
                                 0.0f);
  {
    const int32_t chunk_start = sl + fl + lc;
    std::copy(preds + static_cast<size_t>(chunk_start) * n,
              preds + static_cast<size_t>(chunk_start + chunk_lengths) * n,
              chunk_preds.begin());
  }

  // How much of the logical [FIFO | chunk] leaves the FIFO this step. The
  // chunk_lengths == 0 branch is the finalisation row: no new audio, so flush
  // whatever the FIFO still holds.
  const int32_t combined = fl + chunk_lengths;
  const int32_t overflow = combined - f_cap;
  int32_t pop_out_len = 0;
  if (chunk_lengths == 0) {
    pop_out_len = fl;
  } else if (combined > f_cap) {
    pop_out_len = std::min(combined, std::max(config_.spkcache_update_period,
                                              overflow));
  }
  const int32_t new_fifo_len = combined - pop_out_len;

  // Logical [FIFO(valid) | chunk(central)]. `lc` frames of the chunk are
  // context only: they were fed to the graph for receptive field and must not
  // enter the state, or every frame is counted twice.
  std::vector<float> logical_embs(static_cast<size_t>(combined) * d, 0.0f);
  std::vector<float> logical_preds(static_cast<size_t>(combined) * n, 0.0f);
  std::copy(fifo_.begin(), fifo_.begin() + static_cast<size_t>(fl) * d,
            logical_embs.begin());
  std::copy(current_fifo_preds.begin(),
            current_fifo_preds.begin() + static_cast<size_t>(fl) * n,
            logical_preds.begin());
  {
    const float *src = chunk_embs + static_cast<size_t>(lc) * d;
    std::copy(src, src + static_cast<size_t>(chunk_lengths) * d,
              logical_embs.begin() + static_cast<size_t>(fl) * d);
    std::copy(chunk_preds.begin(),
              chunk_preds.begin() + static_cast<size_t>(chunk_lengths) * n,
              logical_preds.begin() + static_cast<size_t>(fl) * n);
  }

  fifo_.assign(static_cast<size_t>(f_cap) * d, 0.0f);
  fifo_preds_.assign(static_cast<size_t>(f_cap) * n, 0.0f);
  std::copy(logical_embs.begin() + static_cast<size_t>(pop_out_len) * d,
            logical_embs.begin() +
                static_cast<size_t>(pop_out_len + new_fifo_len) * d,
            fifo_.begin());
  std::copy(logical_preds.begin() + static_cast<size_t>(pop_out_len) * n,
            logical_preds.begin() +
                static_cast<size_t>(pop_out_len + new_fifo_len) * n,
            fifo_preds_.begin());
  fifo_lengths_ = new_fifo_len;

  if (pop_out_len > 0) {
    events_.push_back({SortformerCacheEvent::Type::kEviction, pop_out_len});
  }

  // Running silence profile, BEFORE the cache update. Updating the cache first
  // makes compression substitute LAST chunk's silence profile into disabled
  // slots, which drifts slowly and silently.
  if (pop_out_len > 0) {
    std::vector<float> acc(d, 0.0f);
    int64_t sil_count = 0;
    for (int32_t t = 0; t != pop_out_len; ++t) {
      float sum = 0.0f;
      const float *p = logical_preds.data() + static_cast<size_t>(t) * n;
      for (int32_t s = 0; s != n; ++s) {
        sum += p[s];
      }
      if (sum < config_.sil_threshold) {
        ++sil_count;
        const float *e = logical_embs.data() + static_cast<size_t>(t) * d;
        for (int32_t i = 0; i != d; ++i) {
          acc[i] += e[i];
        }
      }
    }
    if (sil_count > 0) {
      for (int32_t i = 0; i != d; ++i) {
        acc[i] += mean_sil_emb_[i] * static_cast<float>(n_sil_frames_);
      }
      n_sil_frames_ += sil_count;
      const float inv = 1.0f / static_cast<float>(std::max<int64_t>(n_sil_frames_, 1));
      for (int32_t i = 0; i != d; ++i) {
        mean_sil_emb_[i] = acc[i] * inv;
      }
    }
  }

  // Append the popped frames to the speaker cache, compressing on overflow.
  const int32_t updated_len = sl + pop_out_len;
  const bool need_compress = updated_len > s_cap;
  const bool first_compression = !spkcache_compressed_ && need_compress;

  // NeMo issue #16002: on the FIRST compression the cached frames must be
  // re-scored with THIS forward's posteriors. The stored ones date from when
  // the cache was empty — the model's least-informed predictions of the session
  // — and the strong boost reserves capacity per speaker column, so a few stale
  // marginal posteriors on an unused slot lock in a phantom speaker.
  const std::vector<float> &old_preds =
      first_compression ? current_spkcache_preds : spkcache_preds_;

  // The candidate buffer is built at FULL width and zero-filled at the tail
  // even when fewer frames arrived: scores_boost_latest is applied at physical
  // offset spkcache_len, so a tightly-packed buffer boosts the wrong rows.
  const int32_t cand_width = s_cap + max_pop_out_len;
  std::vector<float> pad_embs(static_cast<size_t>(cand_width) * d, 0.0f);
  std::vector<float> pad_preds(static_cast<size_t>(cand_width) * n, 0.0f);

  std::copy(spkcache_.begin(), spkcache_.begin() + static_cast<size_t>(sl) * d,
            pad_embs.begin());
  std::copy(old_preds.begin(), old_preds.begin() + static_cast<size_t>(sl) * n,
            pad_preds.begin());
  std::copy(logical_embs.begin(),
            logical_embs.begin() + static_cast<size_t>(pop_out_len) * d,
            pad_embs.begin() + static_cast<size_t>(sl) * d);
  std::copy(logical_preds.begin(),
            logical_preds.begin() + static_cast<size_t>(pop_out_len) * n,
            pad_preds.begin() + static_cast<size_t>(sl) * n);

  if (need_compress) {
    CompressSpkcache(pad_embs, pad_preds, cand_width);
    spkcache_compressed_ = true;
    events_.push_back({SortformerCacheEvent::Type::kCompression, updated_len});
  } else {
    spkcache_.assign(pad_embs.begin(),
                     pad_embs.begin() + static_cast<size_t>(s_cap) * d);
    spkcache_preds_.assign(pad_preds.begin(),
                           pad_preds.begin() + static_cast<size_t>(s_cap) * n);
  }
  spkcache_lengths_ = std::min(updated_len, s_cap);

  chunk_preds.resize(static_cast<size_t>(chunk_lengths) * n);
  return chunk_preds;
}

}  // namespace sherpa_onnx
