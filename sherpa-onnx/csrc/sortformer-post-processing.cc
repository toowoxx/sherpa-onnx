// sherpa-onnx/csrc/sortformer-post-processing.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: see sortformer-post-processing.h for the feature, the four ordered
// steps, and why steps 3 and 4 do not commute.

#include "sherpa-onnx/csrc/sortformer-post-processing.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace sherpa_onnx {

namespace {

struct Region {
  float start;
  float end;
};

// Step 1 + 2: hysteresis binarisation with asymmetric padding, for one speaker
// column. Emitted starts may be negative; clamping happens at the very end so
// the merge in step 3 sees the true distances.
std::vector<Region> Binarize(const std::vector<float> &track, int32_t n_frames,
                             int32_t n_spk, int32_t speaker,
                             float frame_duration,
                             const SortformerPostProcessingConfig &cfg,
                             float clip_duration) {
  std::vector<Region> regions;
  bool active = false;
  float start = 0;

  for (int32_t t = 0; t != n_frames; ++t) {
    const float y = track[static_cast<size_t>(t) * n_spk + speaker];
    const float now = t * frame_duration;
    if (active) {
      if (y < cfg.offset) {
        regions.push_back({start - cfg.pad_onset, now + cfg.pad_offset});
        active = false;
      }
    } else {
      if (y > cfg.onset) {
        start = now;
        active = true;
      }
    }
  }

  if (active) {
    regions.push_back({start - cfg.pad_onset, clip_duration + cfg.pad_offset});
  }

  return regions;
}

// Step 3: merge regions whose gap is below min_duration_off. Overlaps produced
// by the padding have a negative gap and are merged by the same test.
std::vector<Region> Merge(std::vector<Region> regions, float min_duration_off) {
  if (regions.empty()) {
    return regions;
  }

  std::sort(regions.begin(), regions.end(),
            [](const Region &a, const Region &b) { return a.start < b.start; });

  std::vector<Region> merged;
  merged.push_back(regions.front());
  for (size_t i = 1; i != regions.size(); ++i) {
    Region &last = merged.back();
    if (regions[i].start - last.end < min_duration_off) {
      last.end = std::max(last.end, regions[i].end);
    } else {
      merged.push_back(regions[i]);
    }
  }
  return merged;
}

}  // namespace

std::vector<SortformerSegment> SortformerPostProcess(
    const std::vector<float> &track, int32_t n_frames, int32_t n_spk,
    float frame_duration, const SortformerPostProcessingConfig &config,
    float clip_duration) {
  if (clip_duration <= 0) {
    clip_duration = n_frames * frame_duration;
  }

  std::vector<SortformerSegment> out;

  for (int32_t s = 0; s != n_spk; ++s) {
    std::vector<Region> regions = Binarize(track, n_frames, n_spk, s,
                                           frame_duration, config,
                                           clip_duration);
    regions = Merge(std::move(regions), config.min_duration_off);

    for (const Region &r : regions) {
      // Step 4, and only now the clamp: doing it earlier would collapse two
      // distinct sub-zero starts into one and change which regions merged.
      const float start = std::max(0.0f, r.start);
      const float end = std::min(clip_duration, r.end);
      if (end - start < config.min_duration_on) {
        continue;
      }
      out.push_back({s, start, end});
    }
  }

  std::sort(out.begin(), out.end(),
            [](const SortformerSegment &a, const SortformerSegment &b) {
              if (a.start != b.start) {
                return a.start < b.start;
              }
              return a.speaker < b.speaker;
            });

  return out;
}

std::vector<SortformerSegment> SortformerDisjointSegments(
    const std::vector<float> &track, int32_t n_frames, int32_t n_spk,
    float frame_duration, const SortformerPostProcessingConfig &config,
    float clip_duration) {
  if (clip_duration <= 0) {
    clip_duration = n_frames * frame_duration;
  }

  const std::vector<SortformerSegment> overlapping = SortformerPostProcess(
      track, n_frames, n_spk, frame_duration, config, clip_duration);

  // Rasterise the post-processed activity back onto frames. Going through the
  // segment list rather than re-thresholding the track is what makes this the
  // argmax of the POST-PROCESSED activity: a frame the hysteresis, padding and
  // duration filter decided against must not win here.
  std::vector<uint8_t> active(static_cast<size_t>(n_frames) * n_spk, 0);
  for (const SortformerSegment &seg : overlapping) {
    int32_t from = static_cast<int32_t>(std::floor(seg.start / frame_duration));
    int32_t to = static_cast<int32_t>(std::ceil(seg.end / frame_duration));
    from = std::max(0, from);
    to = std::min(n_frames, to);
    for (int32_t t = from; t < to; ++t) {
      active[static_cast<size_t>(t) * n_spk + seg.speaker] = 1;
    }
  }

  constexpr int32_t kSilence = -1;
  std::vector<int32_t> winner(n_frames, kSilence);
  for (int32_t t = 0; t != n_frames; ++t) {
    int32_t best = kSilence;
    float best_p = 0;
    for (int32_t s = 0; s != n_spk; ++s) {
      if (!active[static_cast<size_t>(t) * n_spk + s]) {
        continue;
      }
      const float p = track[static_cast<size_t>(t) * n_spk + s];
      // Strictly greater keeps the LOWER slot index on an exact tie, after the
      // higher posterior has already decided every non-tie.
      if (best == kSilence || p > best_p) {
        best = s;
        best_p = p;
      }
    }
    winner[t] = best;
  }

  std::vector<SortformerSegment> out;
  int32_t t = 0;
  while (t < n_frames) {
    if (winner[t] == kSilence) {
      ++t;
      continue;
    }
    const int32_t s = winner[t];
    int32_t end = t;
    while (end < n_frames && winner[end] == s) {
      ++end;
    }
    out.push_back({s, t * frame_duration,
                   std::min(clip_duration, end * frame_duration)});
    t = end;
  }

  return out;
}

}  // namespace sherpa_onnx
