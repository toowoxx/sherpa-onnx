// sherpa-onnx/csrc/streaming-sortformer-diarization.h
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: the public entry point of the Streaming Sortformer diarization engine —
// create a session, feed a recording's audio, get speaker segments back.
//
// The feature: speaker diarization, which tells who spoke when in a
// recording. The library's other diarizer clusters speaker embeddings, and
// clustering can collapse several distinct speakers into one. This engine
// runs NVIDIA's
// Streaming Sortformer v2.1, which assigns frames to at most four slots
// end-to-end with no clustering step to collapse.
//
// Why the API is session-shaped and not "diarize(wav) -> segments"
//
// Sortformer is streaming-only. There is no batch mode to fall back on: the
// model consumes 10 s chunks and carries a speaker cache forward, so INGESTION
// IS COMPUTE — feeding audio is what runs inference. A one-call API would
// therefore do all its work in one opaque call lasting minutes, and a caller
// that shows progress would have nothing to report. Feeding chunk by chunk is
// what keeps that alive.
//
// The three phases:
//   1. construct  — loads the ONNX session (~469 MB resident; see below)
//   2. AcceptWaveform, repeatedly — buffers audio, computes features, and runs
//      inference for each complete chunk as soon as its right context arrives
//   3. Finalize   — flushes the tail through the model, applies the
//      six-parameter post-processing, returns segments
//
// What the obvious-looking refactor breaks
//
//  * Constructing the ONNX session eagerly and keeping it for the process
//    lifetime looks tidy and is a memory bug: the session is ~469 MB, and a
//    process that keeps it holds that memory for everything else it runs. The
//    session belongs to the diarization run, so the caller must destroy this
//    object when the run ends — including on cancellation and on exception.
//  * The right-context rule is why AcceptWaveform cannot emit a chunk the
//    moment it has chunk_len frames: the chunk is fed to the model WITH one
//    encoder frame of following audio. Emitting early would run every chunk
//    with a truncated receptive field at its right edge, which does not fail,
//    it just quietly degrades. Only Finalize knows a chunk is genuinely last.
//  * Audio is expected at 16 kHz. There is no resampler here on purpose: the
//    call site already reads audio at a chosen rate, and silently resampling
//    would hide a rate mismatch that shifts every timestamp.

#ifndef SHERPA_ONNX_CSRC_STREAMING_SORTFORMER_DIARIZATION_H_
#define SHERPA_ONNX_CSRC_STREAMING_SORTFORMER_DIARIZATION_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/sortformer-diarization-config.h"
#include "sherpa-onnx/csrc/sortformer-post-processing.h"

namespace sherpa_onnx {

class StreamingSortformerDiarization {
 public:
  explicit StreamingSortformerDiarization(
      const SortformerDiarizationConfig &config);

  template <typename Manager>
  StreamingSortformerDiarization(Manager *mgr,
                                 const SortformerDiarizationConfig &config);

  ~StreamingSortformerDiarization();

  StreamingSortformerDiarization(const StreamingSortformerDiarization &) =
      delete;
  StreamingSortformerDiarization &operator=(
      const StreamingSortformerDiarization &) = delete;

  // The only sample rate this engine accepts. Load-bearing: the call site reads
  // audio at whatever this returns, and every emitted timestamp is derived from
  // it.
  int32_t SampleRate() const { return 16000; }

  // Feeds mono float samples in [-1, 1]. Runs inference for every chunk that
  // becomes complete, so this call is where the time goes.
  void AcceptWaveform(const float *samples, int32_t n);

  // Seconds of audio whose diarization result is already committed. A caller's
  // progress numerator; it lags the fed audio by up to one chunk plus its right
  // context, which is inherent and not a rounding error.
  float ProcessedSeconds() const;

  // Seconds of audio accepted so far.
  float AcceptedSeconds() const;

  // Flushes the tail, post-processes and returns DISJOINT segments (one speaker
  // per frame). Idempotent: calling it twice returns the same segments without
  // re-running the model.
  //
  // Disjoint rather than overlap-preserving because a caller that apportions
  // a stretch of time across simultaneous speakers would count its duration
  // twice. The overlap-preserving form is available from
  // SortformerPostProcess on RawTrack() for callers that want it.
  const std::vector<SortformerSegment> &Finalize();

  // The raw per-frame posteriors, (n_frames, n_spk) row-major. Valid after
  // Finalize; before it, only the chunks committed so far.
  const std::vector<float> &RawTrack() const;
  int32_t NumFrames() const;

  void Reset();

  // SHA-256 of the model file, lowercase hex — proof of WHICH artifact ran.
  const std::string &ModelSha256() const;

  // ONNX Runtime intra-op worker thread ids, so a caller can hand the threads
  // that run inference to a platform scheduling API. Populated once the
  // runtime has spawned its pool; empty before the first inference and on
  // platforms that report none.
  const std::vector<int32_t> &WorkerThreadIds() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_STREAMING_SORTFORMER_DIARIZATION_H_
