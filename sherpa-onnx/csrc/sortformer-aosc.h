// sherpa-onnx/csrc/sortformer-aosc.h
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: the Arrival-Order Speaker Cache (AOSC) — the half of Streaming
// Sortformer that is NOT in the exported ONNX graph.
//
// The feature: speaker diarization, which tells who spoke when in a
// recording. The Sortformer engine is a
// streaming neural diarizer, and its exported graph is only a pure function of
// (chunk, speaker cache, FIFO). Everything that makes it a *streaming* diarizer
// — which encoder frames enter the FIFO, when the FIFO overflows into the
// speaker cache, which 188 frames survive compression, what the running silence
// profile is — lives in NeMo's Python `SortformerModules` and is not exported.
// This class is that logic.
//
// Why it is written the way it is
//
// Getting this wrong does not crash and does not throw. It degrades speaker
// separation gradually, and a diarizer that separates speakers badly looks
// the same whether the model or this port is at fault. It is therefore ported
// from executable code rather than from prose: this class is a line-by-line
// port of NeMo's `SortformerModules` as of NVIDIA-NeMo/Speech PR #16032
// (head de724286).
//
// Batch-1 and loop-explicit on purpose. NeMo's implementation is vectorised
// over a ragged batch: every operation is a gather over index tensors built
// from per-row lengths, with a padding row appended so invalid positions have
// somewhere to land. That machinery is entirely about batching. At batch 1
// every one of those gathers collapses to a slice.
//
// What the obvious-looking refactor breaks — each of these is silent
//
//  * The silence profile must be updated BEFORE the speaker cache.
//    `mean_sil_emb` is a running mean over popped frames whose posteriors sum
//    below `sil_threshold`, and compression substitutes it for every disabled
//    cache slot. Update the cache first and compression uses LAST chunk's
//    silence profile, which drifts slowly and invisibly.
//  * `scores_boost_latest` applies to physical positions >= spkcache_len only —
//    the newly appended popped frames in the candidate buffer's PHYSICAL
//    coordinates, not the newest frames in logical order. It is a recency prior
//    on the append region. Applying it "to the newest frames" is a different
//    algorithm.
//  * The strong boost runs before the weak boost AND the weak boost reads the
//    strong boost's output. They are not two independent adjustments; swapping
//    them changes which frames survive.
//  * The compression top-k is over the speaker-major FLATTENED score matrix, so
//    the surviving cache is ordered by speaker and then by original frame
//    index. That ordering is the "arrival order" the cache is named for, and it
//    is what the encoder's positional embedding sees. Sorting the selected flat
//    indices is what produces it — dropping the sort produces a cache that is
//    numerically valid and semantically scrambled.
//  * The candidate buffer is built at width spkcache_len + max_pop_out_len and
//    zero-filled at the tail, even when fewer frames arrived. Compression
//    scores that tail as non-speech and discards it. The padding is not
//    cosmetic: `scores_boost_latest` is applied at PHYSICAL offset
//    spkcache_len, so a tightly-packed buffer boosts the wrong rows.
//  * On the FIRST compression the already-cached frames must be re-scored with
//    THIS forward's posteriors, not the ones stored when they were cached. The
//    stored ones were computed when the cache was empty — the model's
//    least-informed predictions of the whole session — and the strong boost
//    reserves cache capacity per speaker column, so a few stale marginal
//    posteriors on an unused slot lock in a phantom speaker for the rest of the
//    session. This is NeMo issue #16002; the fix is on PR #16032 and is
//    reachable only via the `spkcache_compressed` flag.
//  * chunk_left_context frames are fed to the graph but excluded from the state
//    update. Writing them into the FIFO double-counts every frame.

#ifndef SHERPA_ONNX_CSRC_SORTFORMER_AOSC_H_
#define SHERPA_ONNX_CSRC_SORTFORMER_AOSC_H_

#include <cstdint>
#include <vector>

#include "sherpa-onnx/csrc/sortformer-diarization-config.h"

namespace sherpa_onnx {

// A compression or eviction that actually fired. Recording them lets a test
// assert that both paths were exercised, since a clip too short to reach them
// checks neither.
struct SortformerCacheEvent {
  enum class Type { kEviction, kCompression };
  Type type;
  // Evicted frame count, or the pre-compression candidate length.
  int32_t frames;
};

// One diarization session's AOSC state. Feed encoder chunks in order.
class SortformerAosc {
 public:
  explicit SortformerAosc(const SortformerStreamingConfig &config);

  void Reset();

  // Consumes one chunk's ONNX outputs and advances the state.
  //
  //  chunk_embs        (t_chunk_enc, emb_dim) row-major — the graph's
  //                    `chunk_pre_encode_embs`, INCLUDING context frames.
  //  t_chunk_enc       rows in chunk_embs.
  //  chunk_pre_encode_length  the graph's `chunk_pre_encode_lengths`.
  //  preds             (t_packed, n_spk) row-major — the graph's
  //                    `spkcache_fifo_chunk_preds`, laid out by concat_and_pad
  //                    as [valid spkcache | valid fifo | chunk | padding].
  //  t_packed          rows in preds.
  //  lc, rc            encoder-frame left/right context actually present (0 or
  //                    1; the first chunk has no left context, the last none on
  //                    the right).
  //
  // Returns this chunk's contribution to the output track, (n, n_spk)
  // row-major with context frames already stripped.
  std::vector<float> Step(const float *chunk_embs, int32_t t_chunk_enc,
                          int32_t chunk_pre_encode_length, const float *preds,
                          int32_t t_packed, int32_t lc, int32_t rc);

  // Graph inputs for the next step. spkcache is (spkcache_len, emb_dim) and
  // fifo is (fifo_len, emb_dim), both row-major and always at full capacity —
  // the zero padding beyond the valid lengths is provably free (measured max
  // abs diff 0.0 between a capacity-padded and a tightly-packed concat).
  const std::vector<float> &Spkcache() const { return spkcache_; }
  const std::vector<float> &Fifo() const { return fifo_; }
  int32_t SpkcacheLengths() const { return spkcache_lengths_; }
  int32_t FifoLengths() const { return fifo_lengths_; }

  const std::vector<SortformerCacheEvent> &Events() const { return events_; }

 private:
  // NeMo _get_log_pred_scores: high for confident NON-overlapped speech.
  std::vector<float> LogPredScores(const float *preds, int32_t n_frames) const;

  // NeMo _disable_low_scores: -inf for non-speech, and for overlapped speech
  // once a speaker already has enough confidently-solo frames.
  void DisableLowScores(const float *preds, int32_t n_frames,
                        std::vector<float> *scores) const;

  // NeMo _boost_topk_scores. In place and order-dependent.
  void BoostTopK(int32_t n_frames, int32_t n_per_spk, float scale_factor,
                 std::vector<float> *scores) const;

  // NeMo _compress_spkcache: keep the spkcache_len most important frames,
  // ordered by speaker then by original frame index.
  void CompressSpkcache(const std::vector<float> &embs,
                        const std::vector<float> &preds, int32_t n_frames);

  SortformerStreamingConfig config_;

  std::vector<float> spkcache_;        // (spkcache_len, emb_dim)
  std::vector<float> spkcache_preds_;  // (spkcache_len, n_spk)
  int32_t spkcache_lengths_ = 0;
  bool spkcache_compressed_ = false;

  std::vector<float> fifo_;        // (fifo_len, emb_dim)
  std::vector<float> fifo_preds_;  // (fifo_len, n_spk)
  int32_t fifo_lengths_ = 0;

  std::vector<float> mean_sil_emb_;  // (emb_dim)
  int64_t n_sil_frames_ = 0;

  std::vector<SortformerCacheEvent> events_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SORTFORMER_AOSC_H_
