// sherpa-onnx/jni/streaming-sortformer-diarization.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: the JNI surface of the Streaming Sortformer diarization engine.
//
// The feature: speaker diarization, which tells who spoke when in a
// recording. The engine
// itself is C++ (see csrc/streaming-sortformer-diarization.h); this is how
// Kotlin code runs it. It is deliberately the SESSION shape rather
// than a one-call "diarize this file", because Sortformer computes as it
// ingests and a caller can show progress while it runs.
//
// Three calls exist here to show which model ran and how, and have
// no counterpart in the other engines:
//
//  * modelSha256 — different model files can occupy the SAME path, so a log
//    line naming the file does not identify what ran. The digest does.
//  * workerThreadIds — the ONNX Runtime worker thread ids, so a caller can
//    hand the threads that run inference to a platform scheduling API, which
//    needs exactly those ids. This returns the ids and nothing else; what to
//    do with them is the Kotlin caller's decision.
//  * rawTrack / numFrames — the raw posterior track, which lets a caller
//    compare the posteriors element by element with a reference. Nothing
//    else on this surface carries them: `finalizeSession`'s segments have
//    been through argmax and the six-parameter hysteresis, so comparing them
//    would pass or fail for reasons unrelated to the posteriors, and a small
//    divergence that post-processing absorbs would go unseen.
//
// Ownership: newFromFile/newFromAsset return a raw pointer the Kotlin side owns
// and MUST release in a finally block. The session holds a ~469 MB ONNX
// session, which is why leaking one is not a slow leak but an immediate OOM on
// the next diarization run.

#include "sherpa-onnx/csrc/streaming-sortformer-diarization.h"

#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static SortformerDiarizationConfig GetSortformerDiarizationConfig(
    JNIEnv *env, jobject config, bool *ok) {
  SortformerDiarizationConfig ans;

  jclass cls = env->GetObjectClass(config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model, model, cls, config);
  SHERPA_ONNX_JNI_READ_INT(ans.num_threads, numThreads, cls, config);
  SHERPA_ONNX_JNI_READ_BOOL(ans.debug, debug, cls, config);
  SHERPA_ONNX_JNI_READ_STRING(ans.provider, provider, cls, config);

  *ok = true;
  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  bool ok = false;
  auto config = sherpa_onnx::GetSortformerDiarizationConfig(env, _config, &ok);
  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in the Sortformer config!");
    return 0;
  }

  auto engine = new sherpa_onnx::StreamingSortformerDiarization(config);
  return (jlong)engine;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  bool ok = false;
  auto config = sherpa_onnx::GetSortformerDiarizationConfig(env, _config, &ok);
  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  auto engine = new sherpa_onnx::StreamingSortformerDiarization(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);
  return (jlong)engine;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::StreamingSortformerDiarization *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples) {
  auto engine =
      reinterpret_cast<sherpa_onnx::StreamingSortformerDiarization *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  engine->AcceptWaveform(p, n);
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jfloat JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_processedSeconds(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_onnx::StreamingSortformerDiarization *>(ptr)
      ->ProcessedSeconds();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jfloat JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_acceptedSeconds(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_onnx::StreamingSortformerDiarization *>(ptr)
      ->AcceptedSeconds();
}

// Segments are returned FLATTENED as [speaker, start, end, speaker, start, ...]
// rather than as an array of objects: constructing N Java objects across JNI
// costs a class lookup and a constructor call each, and the Kotlin side has to
// walk the result once anyway to build its own domain type.
SHERPA_ONNX_EXTERN_C
JNIEXPORT jfloatArray JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_finalizeSession(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto engine =
      reinterpret_cast<sherpa_onnx::StreamingSortformerDiarization *>(ptr);
  const auto &segments = engine->Finalize();

  std::vector<float> flat;
  flat.reserve(segments.size() * 3);
  for (const auto &s : segments) {
    flat.push_back(static_cast<float>(s.speaker));
    flat.push_back(s.start);
    flat.push_back(s.end);
  }

  jfloatArray out = env->NewFloatArray(flat.size());
  env->SetFloatArrayRegion(out, 0, flat.size(), flat.data());
  return out;
}

// The posterior track is returned FLAT — (n_frames * n_spk) row-major — with
// numFrames supplying the row count. A jobjectArray of per-frame float[] would
// allocate one Java array per frame (750 of them per minute of audio) for a
// track the caller walks exactly once.
SHERPA_ONNX_EXTERN_C
JNIEXPORT jfloatArray JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_rawTrack(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto engine =
      reinterpret_cast<sherpa_onnx::StreamingSortformerDiarization *>(ptr);
  const std::vector<float> &track = engine->RawTrack();

  jfloatArray out = env->NewFloatArray(track.size());
  env->SetFloatArrayRegion(out, 0, track.size(), track.data());
  return out;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_numFrames(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_onnx::StreamingSortformerDiarization *>(ptr)
      ->NumFrames();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_modelSha256(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto engine =
      reinterpret_cast<sherpa_onnx::StreamingSortformerDiarization *>(ptr);
  return env->NewStringUTF(engine->ModelSha256().c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jintArray JNICALL
Java_com_k2fsa_sherpa_onnx_StreamingSortformerDiarization_workerThreadIds(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto engine =
      reinterpret_cast<sherpa_onnx::StreamingSortformerDiarization *>(ptr);
  const std::vector<int32_t> &tids = engine->WorkerThreadIds();

  jintArray out = env->NewIntArray(tids.size());
  env->SetIntArrayRegion(out, 0, tids.size(),
                         reinterpret_cast<const jint *>(tids.data()));
  return out;
}
