// sherpa-onnx/kotlin-api/StreamingSortformerDiarization.kt
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: the Kotlin binding for the Streaming Sortformer diarization engine.
//
// The feature: speaker diarization, which tells who spoke when in a
// recording. The library's other diarizer clusters speaker embeddings, and
// clustering can collapse several distinct speakers into one. This binds
// NVIDIA's
// Streaming Sortformer v2.1, which assigns frames to at most four slots
// end-to-end with no clustering step to collapse.
//
// Why the API is a session and not `diarize(audio): List<Segment>`
//
// Sortformer has no batch mode. It consumes fixed chunks and carries a speaker
// cache forward, so FEEDING AUDIO IS WHAT RUNS INFERENCE. A single-call API
// would spend minutes inside one opaque call and a caller's progress display
// would have nothing to report. Feed audio in pieces, read `processedSeconds`
// for progress, call `finalizeSession` once at the end.
//
// What the obvious-looking refactor breaks
//
//  * `release()` is NOT optional and NOT best-effort. The native side holds an
//    ONNX session of roughly 469 MB. Relying on `finalize()` defers the
//    free to an arbitrary GC and turns a leak into an OOM on the next run.
//    Call it in a `finally`.
//  * `processedSeconds` lags the audio fed by up to one chunk plus its right
//    context. That is inherent — a chunk cannot run before the audio that
//    follows it exists — and is not a rounding error to "fix" by reporting
//    accepted audio instead.
//  * `speakerId` is a SLOT index within one session, not a stable identity.
//    Persisting it, or comparing it across sessions, is meaningless.
//  * `rawTrack`/`numFrames`/`modelSha256`/`workerThreadIds` look like dead
//    surface to a caller that only wants segments, and are not: they show
//    which model ran and what it computed. `rawTrack` lets a caller compare
//    the posteriors element by element with a reference, and no other member
//    carries them: `finalizeSession` returns segments taken through argmax
//    and hysteresis. Deleting them as unused would remove the only way to
//    check the engine's posteriors against a reference.

package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

data class SortformerDiarizationConfig(
    val model: String,
    val numThreads: Int = 2,
    val debug: Boolean = false,
    val provider: String = "cpu",
)

/**
 * One diarized region. [speakerId] is a slot index in 0..3, valid only within
 * the session that produced it.
 */
data class SortformerSegment(
    val speakerId: Int,
    val startSeconds: Float,
    val endSeconds: Float,
)

class StreamingSortformerDiarization(
    assetManager: AssetManager? = null,
    config: SortformerDiarizationConfig,
) {
    private var ptr: Long

    init {
        ptr = if (assetManager != null) {
            newFromAsset(assetManager, config)
        } else {
            newFromFile(config)
        }
        require(ptr != 0L) {
            "Invalid SortformerDiarizationConfig: failed to create the native Sortformer session"
        }
    }

    /** The only sample rate this engine accepts. Every timestamp derives from it. */
    val sampleRate: Int = 16000

    /**
     * Feeds mono samples in [-1, 1]. This is where the time goes: any chunk that
     * becomes complete is run through the model during this call.
     */
    fun acceptWaveform(samples: FloatArray) = acceptWaveform(ptr, samples)

    /** Seconds of audio whose diarization result is already committed. */
    fun processedSeconds(): Float = processedSeconds(ptr)

    /** Seconds of audio accepted so far. */
    fun acceptedSeconds(): Float = acceptedSeconds(ptr)

    /**
     * Flushes the tail, post-processes and returns disjoint segments. Idempotent.
     */
    fun finalizeSession(): List<SortformerSegment> {
        // Flattened as [speaker, start, end, ...] on the native side: building N
        // Java objects across JNI costs a class lookup and a constructor call
        // each, and this loop has to run anyway.
        val flat = finalizeSession(ptr)
        return (flat.indices step 3).map { i ->
            SortformerSegment(
                speakerId = flat[i].toInt(),
                startSeconds = flat[i + 1],
                endSeconds = flat[i + 2],
            )
        }
    }

    /**
     * SHA-256 of the model file that this session actually loaded, lowercase hex.
     *
     * Exists because different model files can occupy the same path, so a
     * log line naming the file does not say what ran. The digest does.
     */
    fun modelSha256(): String = modelSha256(ptr)

    /**
     * ONNX Runtime intra-op worker thread ids.
     *
     * Lets a caller hand the threads that run inference to a platform
     * scheduling API, which needs exactly these ids. Empty until the runtime
     * has spawned its pool, i.e. before the first [acceptWaveform] that runs a
     * chunk.
     */
    fun workerThreadIds(): IntArray = workerThreadIds(ptr)

    /**
     * The raw per-frame posteriors, FLAT and row-major: [numFrames] rows of four
     * sigmoid elements, element `s` of frame `f` at `rawTrack()[f * 4 + s]`.
     *
     * Lets a caller compare the posteriors element by element with a
     * reference. [finalizeSession]'s segments cannot stand in for this:
     * they have already been through argmax and the six-parameter hysteresis, so
     * a segment-level comparison passes or fails for reasons unrelated to the
     * posteriors, and a small divergence that post-processing absorbs stays
     * invisible.
     *
     * Valid after [finalizeSession] — before it, only the chunks committed so
     * far, and the tail is not yet trimmed to the frames the audio covers.
     */
    fun rawTrack(): FloatArray = rawTrack(ptr)

    /** Row count of [rawTrack] — its length divided by the four slots. */
    fun numFrames(): Int = numFrames(ptr)

    protected fun finalize() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    /** Frees the native session. Call this in a `finally` — see the SPEC. */
    fun release() = finalize()

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: SortformerDiarizationConfig,
    ): Long

    private external fun newFromFile(config: SortformerDiarizationConfig): Long

    private external fun delete(ptr: Long)

    private external fun acceptWaveform(ptr: Long, samples: FloatArray)

    private external fun processedSeconds(ptr: Long): Float

    private external fun acceptedSeconds(ptr: Long): Float

    private external fun finalizeSession(ptr: Long): FloatArray

    private external fun modelSha256(ptr: Long): String

    private external fun workerThreadIds(ptr: Long): IntArray

    private external fun rawTrack(ptr: Long): FloatArray

    private external fun numFrames(ptr: Long): Int

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
