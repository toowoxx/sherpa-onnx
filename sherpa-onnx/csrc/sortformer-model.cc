// sherpa-onnx/csrc/sortformer-model.cc
//
// Copyright (c)  2026  Toowoxx IT GmbH
//
// SPEC: see sortformer-model.h for the feature, the graph's I/O contract and
// why the state tensors are passed at full capacity.

#include "sherpa-onnx/csrc/sortformer-model.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// SPEC: SHA-256 (FIPS 180-4), self-contained because sherpa-onnx has no crypto
// dependency and pulling one in for a single integrity check is not worth the
// build-matrix cost across Android, iOS, WASM and four desktop platforms.
//
// It exists for one reason: different model files can occupy the SAME path,
// so "the model at path X" does not identify what actually ran. The digest
// identifies the file a session loaded, and a path in a log line cannot do
// that.
class Sha256 {
 public:
  void Update(const uint8_t *data, size_t len) {
    total_ += len;
    while (len > 0) {
      const size_t take = std::min(len, sizeof(buf_) - buf_len_);
      std::memcpy(buf_ + buf_len_, data, take);
      buf_len_ += take;
      data += take;
      len -= take;
      if (buf_len_ == sizeof(buf_)) {
        Transform(buf_);
        buf_len_ = 0;
      }
    }
  }

  std::string HexDigest() {
    const uint64_t bit_len = total_ * 8;
    uint8_t pad = 0x80;
    Update(&pad, 1);
    total_ -= 1;  // padding is not message length
    pad = 0x00;
    while (buf_len_ != 56) {
      Update(&pad, 1);
      total_ -= 1;
    }
    uint8_t len_be[8];
    for (int i = 0; i != 8; ++i) {
      len_be[i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));
    }
    Update(len_be, 8);

    std::string out;
    out.reserve(64);
    char tmp[3];
    for (uint32_t h : h_) {
      for (int i = 3; i >= 0; --i) {
        snprintf(tmp, sizeof(tmp), "%02x",
                 static_cast<unsigned>((h >> (8 * i)) & 0xff));
        out += tmp;
      }
    }
    return out;
  }

 private:
  static uint32_t Ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void Transform(const uint8_t *block) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

    uint32_t w[64];
    for (int i = 0; i != 16; ++i) {
      w[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
             (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
             (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
             static_cast<uint32_t>(block[4 * i + 3]);
    }
    for (int i = 16; i != 64; ++i) {
      const uint32_t s0 = Ror(w[i - 15], 7) ^ Ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Ror(w[i - 2], 17) ^ Ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
    for (int i = 0; i != 64; ++i) {
      const uint32_t s1 = Ror(e, 6) ^ Ror(e, 11) ^ Ror(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      const uint32_t s0 = Ror(a, 2) ^ Ror(a, 13) ^ Ror(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
  }

  uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buf_[64] = {};
  size_t buf_len_ = 0;
  uint64_t total_ = 0;
};

std::string Sha256Hex(const uint8_t *data, size_t len) {
  Sha256 h;
  h.Update(data, len);
  return h.HexDigest();
}

std::string Sha256HexOfFile(const std::string &path) {
  FILE *f = fopen(path.c_str(), "rb");
  if (f == nullptr) {
    return {};
  }
  Sha256 h;
  std::vector<uint8_t> buf(1 << 20);
  size_t n = 0;
  while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) {
    h.Update(buf.data(), n);
  }
  fclose(f);
  return h.HexDigest();
}

int32_t CurrentThreadId() {
#if defined(__linux__)
  return static_cast<int32_t>(syscall(SYS_gettid));
#else
  return 0;
#endif
}

// SPEC (worker thread ids): ONNX Runtime does not expose its intra-op worker
// thread ids, and a caller that hands the threads running inference to a
// platform scheduling API needs exactly those ids. A request made for the
// wrong threads leaves the inference threads unaffected.
//
// The supported way in is SessionOptions::SetCustomCreateThreadFn: the runtime
// asks US to spawn each worker, so we record the tid inside the new thread
// before handing control to the runtime's worker function.
//
// This is INERT here by design. Nothing in this file consumes the ids; a
// caller reads them and decides what to do with them. Calling a platform
// scheduling API here would tie the library to that platform and would apply
// the call to every caller.
struct WorkerThreadRegistry {
  std::mutex mutex;
  std::vector<int32_t> tids;
};

struct WorkerThreadHandle {
  std::thread thread;
};

// The handle type is OrtCustomHandleType*, an opaque INCOMPLETE type. It is an
// opaque cookie ORT only ever hands back to our join function, so reinterpreting
// our own pointer through it is the intended usage; returning void* does not
// compile.
const OrtCustomHandleType *CreateWorkerThread(void *options,
                                              OrtThreadWorkerFn worker_fn,
                                              void *worker_param) {
  auto *registry = reinterpret_cast<WorkerThreadRegistry *>(options);
  auto *handle = new WorkerThreadHandle();
  handle->thread = std::thread([registry, worker_fn, worker_param]() {
    if (registry != nullptr) {
      std::lock_guard<std::mutex> lock(registry->mutex);
      registry->tids.push_back(CurrentThreadId());
    }
    worker_fn(worker_param);
  });
  return reinterpret_cast<const OrtCustomHandleType *>(handle);
}

void JoinWorkerThread(const OrtCustomHandleType *handle_ptr) {
  auto *handle = const_cast<WorkerThreadHandle *>(
      reinterpret_cast<const WorkerThreadHandle *>(handle_ptr));
  if (handle == nullptr) {
    return;
  }
  if (handle->thread.joinable()) {
    handle->thread.join();
  }
  delete handle;
}

}  // namespace

class SortformerModel::Impl {
 public:
  explicit Impl(const SortformerDiarizationConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptionsImpl(config.num_threads, config.provider)),
        allocator_{} {
    InstallThreadHooks();
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.model), sess_opts_);
    model_sha256_ = Sha256HexOfFile(config_.model);
    Init();
  }

  template <typename Manager>
  Impl(Manager *mgr, const SortformerDiarizationConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptionsImpl(config.num_threads, config.provider)),
        allocator_{} {
    InstallThreadHooks();
    auto buf = ReadFile(mgr, config_.model);
    model_sha256_ = Sha256Hex(reinterpret_cast<const uint8_t *>(buf.data()),
                              buf.size());
    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                           sess_opts_);
    Init();
  }

  SortformerModelOutput Forward(const float *features, int32_t n_feat_frames,
                                const std::vector<float> &spkcache,
                                int32_t spkcache_lengths,
                                const std::vector<float> &fifo,
                                int32_t fifo_lengths) const {
    const auto &s = config_.streaming;
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> chunk_shape{1, n_feat_frames, s.feat_dim};
    std::array<int64_t, 1> scalar_shape{1};
    std::array<int64_t, 3> spkcache_shape{1, s.spkcache_len, s.emb_dim};
    std::array<int64_t, 3> fifo_shape{1, s.fifo_len, s.emb_dim};

    int64_t chunk_len_value = n_feat_frames;
    int64_t spkcache_len_value = spkcache_lengths;
    int64_t fifo_len_value = fifo_lengths;

    std::vector<Ort::Value> inputs;
    inputs.reserve(6);
    inputs.push_back(Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(features),
        static_cast<size_t>(n_feat_frames) * s.feat_dim, chunk_shape.data(),
        chunk_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor(memory_info, &chunk_len_value, 1,
                                              scalar_shape.data(),
                                              scalar_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(spkcache.data()), spkcache.size(),
        spkcache_shape.data(), spkcache_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor(memory_info, &spkcache_len_value,
                                              1, scalar_shape.data(),
                                              scalar_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(fifo.data()), fifo.size(),
        fifo_shape.data(), fifo_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor(memory_info, &fifo_len_value, 1,
                                              scalar_shape.data(),
                                              scalar_shape.size()));

    // Inputs are ordered to match input_names_, which was read from the graph
    // rather than assumed; the export names its tensors after internal ops, so
    // positional assumptions are not safe here.
    std::vector<const char *> ordered_input_names;
    std::vector<Ort::Value> ordered_inputs;
    ordered_input_names.reserve(6);
    ordered_inputs.reserve(6);
    for (const std::string &name : input_names_) {
      const int32_t index = InputIndex(name);
      if (index < 0) {
        SHERPA_ONNX_LOGE(
            "sortformer: the model expects an input named '%s', which this "
            "build does not know how to provide. The export contract in "
            "sortformer-model.h and the artifact have diverged.",
            name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      ordered_input_names.push_back(name.c_str());
      ordered_inputs.push_back(std::move(inputs[index]));
    }

    auto out = sess_->Run({}, ordered_input_names.data(), ordered_inputs.data(),
                          ordered_inputs.size(), output_names_ptr_.data(),
                          output_names_ptr_.size());

    SortformerModelOutput result;
    {
      const Ort::Value &preds = out[preds_index_];
      auto shape = preds.GetTensorTypeAndShapeInfo().GetShape();
      result.n_packed = static_cast<int32_t>(shape[shape.size() - 2]);
      const int32_t n_spk = static_cast<int32_t>(shape[shape.size() - 1]);
      const float *data = preds.GetTensorData<float>();
      result.preds.assign(data,
                          data + static_cast<size_t>(result.n_packed) * n_spk);
    }
    {
      const Ort::Value &embs = out[embs_index_];
      auto shape = embs.GetTensorTypeAndShapeInfo().GetShape();
      result.n_chunk_enc = static_cast<int32_t>(shape[shape.size() - 2]);
      const int32_t dim = static_cast<int32_t>(shape[shape.size() - 1]);
      const float *data = embs.GetTensorData<float>();
      result.chunk_pre_encode_embs.assign(
          data, data + static_cast<size_t>(result.n_chunk_enc) * dim);
    }
    result.chunk_pre_encode_length =
        static_cast<int32_t>(out[enc_len_index_].GetTensorData<int64_t>()[0]);

    return result;
  }

  const std::string &ModelSha256() const { return model_sha256_; }

  const std::vector<int32_t> &WorkerThreadIds() const {
    std::lock_guard<std::mutex> lock(thread_registry_->mutex);
    cached_tids_ = thread_registry_->tids;
    return cached_tids_;
  }

 private:
  void InstallThreadHooks() {
    thread_registry_ = std::make_shared<WorkerThreadRegistry>();
    sess_opts_.SetCustomCreateThreadFn(CreateWorkerThread);
    sess_opts_.SetCustomThreadCreationOptions(thread_registry_.get());
    sess_opts_.SetCustomJoinThreadFn(JoinWorkerThread);
  }

  int32_t InputIndex(const std::string &name) const {
    if (name == "chunk") return 0;
    if (name == "chunk_lengths") return 1;
    if (name == "spkcache") return 2;
    if (name == "spkcache_lengths") return 3;
    if (name == "fifo") return 4;
    if (name == "fifo_lengths") return 5;
    return -1;
  }

  void Init() {
    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // Outputs are located by NAME. The export names two of the three after the
    // op that produced them, which is a strong sign the ordering is incidental.
    for (size_t i = 0; i != output_names_.size(); ++i) {
      const std::string &n = output_names_[i];
      if (n.find("preds") != std::string::npos) {
        preds_index_ = static_cast<int32_t>(i);
      } else if (n.find("lengths") != std::string::npos) {
        enc_len_index_ = static_cast<int32_t>(i);
      } else if (n.find("embs") != std::string::npos) {
        embs_index_ = static_cast<int32_t>(i);
      }
    }

    if (preds_index_ < 0 || embs_index_ < 0 || enc_len_index_ < 0) {
      SHERPA_ONNX_LOGE(
          "sortformer: could not locate the three expected outputs by name. "
          "Found %d outputs; the artifact does not match the export contract "
          "in sortformer-model.h.",
          static_cast<int32_t>(output_names_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("sortformer model: sha256=%s", model_sha256_.c_str());
    }
  }

  SortformerDiarizationConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t preds_index_ = -1;
  int32_t embs_index_ = -1;
  int32_t enc_len_index_ = -1;

  std::string model_sha256_;
  std::shared_ptr<WorkerThreadRegistry> thread_registry_;
  mutable std::vector<int32_t> cached_tids_;
};

SortformerModel::SortformerModel(const SortformerDiarizationConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
SortformerModel::SortformerModel(Manager *mgr,
                                 const SortformerDiarizationConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

SortformerModel::~SortformerModel() = default;

SortformerModelOutput SortformerModel::Forward(
    const float *features, int32_t n_feat_frames,
    const std::vector<float> &spkcache, int32_t spkcache_lengths,
    const std::vector<float> &fifo, int32_t fifo_lengths) const {
  return impl_->Forward(features, n_feat_frames, spkcache, spkcache_lengths,
                        fifo, fifo_lengths);
}

const std::string &SortformerModel::ModelSha256() const {
  return impl_->ModelSha256();
}

const std::vector<int32_t> &SortformerModel::WorkerThreadIds() const {
  return impl_->WorkerThreadIds();
}

#if __ANDROID_API__ >= 9
template SortformerModel::SortformerModel(
    AAssetManager *mgr, const SortformerDiarizationConfig &config);
#endif

#if __OHOS__
template SortformerModel::SortformerModel(
    NativeResourceManager *mgr, const SortformerDiarizationConfig &config);
#endif

}  // namespace sherpa_onnx
