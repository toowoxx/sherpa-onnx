// sherpa-onnx/csrc/sortformer-post-processing.h
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: turns Sortformer's per-frame speaker posteriors into the timed speaker
// segments a caller consumes.
//
// The feature: speaker diarization, which tells who spoke when. The Sortformer
// engine produces a 4-column posterior track at one value per 80 ms frame;
// a caller needs "speaker 2 spoke from 12.4s to 19.8s". This is that
// conversion.
//
// Why it is not a threshold at 0.5
//
// Naive thresholding underperforms the published numbers noticeably. NVIDIA
// ships a tuned post-processing configuration with the checkpoint and the
// published accuracy is measured WITH it, so the port has to reproduce it or
// it would run a handicapped model. Concretely it is:
//
//   1. Hysteresis binarisation: a speaker turns ON when the posterior exceeds
//      `onset` (0.641) and stays on until it falls below `offset` (0.561). Two
//      thresholds, not one — a single threshold chops a continuous utterance
//      into fragments every time the posterior wobbles across it, which is what
//      inflates turn counts.
//   2. Padding: each region is extended by `pad_onset` (0.229 s) to the left
//      and `pad_offset` (0.079 s) to the right. Speech onsets are detected late
//      and offsets early, so the padding recovers real audio; it is asymmetric
//      because the two errors are not.
//   3. Merging: regions of the SAME speaker separated by less than
//      `min_duration_off` (0.296 s) become one. This also absorbs the overlaps
//      that step 2 creates.
//   4. Duration filtering: segments shorter than `min_duration_on` (0.511 s)
//      are dropped.
//
// There is NO median filter. Upstream's
// diar_streaming_sortformer_4spk-v2_callhome-part1.yaml has exactly the six
// parameters above and NeMo's predlist_to_timestamps applies only the four
// steps above. Adding a median filter would move every boundary away from the
// segments NeMo produces.
//
// What the obvious-looking refactor breaks
//
//  * Steps 3 and 4 do not commute. Merging first can rescue two 0.3 s
//    fragments 0.1 s apart into one 0.7 s segment that survives filtering;
//    filtering first deletes both. NeMo merges first.
//  * The order of the padding and the merge matters for the same reason: the
//    merge sees the PADDED regions, so a gap of 0.5 s between raw regions can
//    already be closed by padding before min_duration_off is consulted.
//  * `pad_onset` can push a segment's start below zero. Clamp at emission, not
//    during the merge — clamping early turns two distinct sub-zero starts into
//    one and changes which regions merge.
//
// Domain caveat: this is a per-speaker operation. Sortformer emits overlapping
// speech naturally (two columns hot at once), and this file preserves that.
// Reducing the result to DISJOINT segments is the caller's job and deliberately
// out of scope here. SortformerDisjointSegments below does it with a per-frame
// argmax, for a caller that would count the duration of simultaneous speech
// twice.

#ifndef SHERPA_ONNX_CSRC_SORTFORMER_POST_PROCESSING_H_
#define SHERPA_ONNX_CSRC_SORTFORMER_POST_PROCESSING_H_

#include <cstdint>
#include <vector>

#include "sherpa-onnx/csrc/sortformer-diarization-config.h"

namespace sherpa_onnx {

struct SortformerSegment {
  // Zero-based slot index in [0, n_spk). NOT a stable speaker identity across
  // sessions — the model assigns slots in arrival order within one session.
  int32_t speaker = 0;
  float start = 0;  // seconds
  float end = 0;    // seconds
};

// `track` is (n_frames, n_spk) row-major posteriors; `frame_duration` is
// seconds per frame (0.08 at the default geometry). `clip_duration` bounds the
// last segment; pass <= 0 to bound it at n_frames * frame_duration.
//
// The result is sorted by (start, speaker) and may contain overlapping
// segments belonging to different speakers.
std::vector<SortformerSegment> SortformerPostProcess(
    const std::vector<float> &track, int32_t n_frames, int32_t n_spk,
    float frame_duration, const SortformerPostProcessingConfig &config,
    float clip_duration = 0);

// Per-frame argmax over the POST-PROCESSED activity, yielding disjoint
// segments — one speaker per frame, silence where no speaker is active. Ties
// go to the higher raw posterior, then to the lower slot index.
//
// Separate from SortformerPostProcess because the overlap-preserving form is
// the model's real output and the disjoint form is a lossy projection some
// callers need; conflating them would hide which one a caller is getting.
std::vector<SortformerSegment> SortformerDisjointSegments(
    const std::vector<float> &track, int32_t n_frames, int32_t n_spk,
    float frame_duration, const SortformerPostProcessingConfig &config,
    float clip_duration = 0);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SORTFORMER_POST_PROCESSING_H_
