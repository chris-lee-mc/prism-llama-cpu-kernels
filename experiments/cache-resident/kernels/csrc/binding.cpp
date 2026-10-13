// pybind11 glue: raw device pointers and stream handles as int64, so callers can launch inside CUDA graph capture.
#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "ternary_gemv.h"
#include "toy_ops.h"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME cr_ternary_ext
#endif

namespace py = pybind11;

namespace {

template <typename T>
T* ptr(int64_t v) {
  return reinterpret_cast<T*>(static_cast<intptr_t>(v));
}

cudaStream_t stream_of(int64_t v) { return reinterpret_cast<cudaStream_t>(static_cast<intptr_t>(v)); }

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void gemv(int variant, int64_t q, int64_t s, int64_t x, int64_t y, int N, int K, int64_t stream) {
  if (variant == 0) {
    check(cr_gemv_lut(ptr<const uint32_t>(q), ptr<const __half>(s), ptr<const __half>(x), ptr<__half>(y), N, K, stream_of(stream)), "cr_gemv_lut");
  } else if (variant == 1) {
    check(cr_gemv_fp4(ptr<const uint32_t>(q), ptr<const __half>(s), ptr<const __half>(x), ptr<__half>(y), N, K, stream_of(stream)), "cr_gemv_fp4");
  } else {
    throw std::runtime_error("gemv: variant must be 0 (lut) or 1 (fp4)");
  }
}

void gdn_step(int64_t S, int64_t qkv, int64_t ab, int64_t o, int H, int dk, int dv, int64_t stream) {
  check(cr_gdn_step(ptr<float>(S), ptr<const __half>(qkv), ptr<const __half>(ab), ptr<__half>(o), H, dk, dv, stream_of(stream)), "cr_gdn_step");
}

void attn_decode(int64_t q, int64_t Kc, int64_t Vc, int64_t o, int Hq, int Hkv, int D, int T, float scale, int64_t stream) {
  check(cr_attn_decode(ptr<const __half>(q), ptr<const __half>(Kc), ptr<const __half>(Vc), ptr<__half>(o), Hq, Hkv, D, T, scale, stream_of(stream)),
        "cr_attn_decode");
}

void rmsnorm(int64_t x, int64_t w, int64_t y, int n, float eps, int64_t stream) {
  check(cr_rmsnorm(ptr<const __half>(x), ptr<const __half>(w), ptr<__half>(y), n, eps, stream_of(stream)), "cr_rmsnorm");
}

void silu_mul(int64_t g, int64_t u, int64_t y, int n, int64_t stream) {
  check(cr_silu_mul(ptr<const __half>(g), ptr<const __half>(u), ptr<__half>(y), n, stream_of(stream)), "cr_silu_mul");
}

void add_(int64_t a, int64_t b, int n, int64_t stream) {
  check(cr_add(ptr<__half>(a), ptr<const __half>(b), n, stream_of(stream)), "cr_add");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "cache-resident ternary GEMV and toy decode ops (raw pointer API)";
  m.def("gemv", &gemv, "variant 0=lut 1=fp4; q,s,x,y device pointers");
  m.def("gdn_step", &gdn_step);
  m.def("attn_decode", &attn_decode);
  m.def("rmsnorm", &rmsnorm);
  m.def("silu_mul", &silu_mul);
  m.def("add_", &add_);
}
