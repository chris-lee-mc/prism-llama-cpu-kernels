// Decode profiling target for E4: fill the context to a depth, then decode n tokens between
// cudaProfilerStart/Stop so ncu (--profile-from-start off) sees only decode kernels.
// Build: see exps/e4_bonsai.py build_prof_tool(). Prints one JSON line with wall timing.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" int cudaProfilerStart(void);
extern "C" int cudaProfilerStop(void);

static ggml_type parse_kv(const char * s) {
    if (!strcmp(s, "f16")) return GGML_TYPE_F16;
    if (!strcmp(s, "q8_0")) return GGML_TYPE_Q8_0;
    fprintf(stderr, "unsupported kv type %s (use f16 or q8_0)\n", s);
    exit(2);
}

int main(int argc, char ** argv) {
    std::string model;
    int depth = 0, n_tok = 4, n_ub = 2048, n_gpu = 99, n_warm = 2;
    bool fa = true, prof = true;
    ggml_type kv = GGML_TYPE_F16;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "-m") model = next();
        else if (a == "-d") depth = atoi(next());
        else if (a == "-n") n_tok = atoi(next());
        else if (a == "-ub") n_ub = atoi(next());
        else if (a == "-ngl") n_gpu = atoi(next());
        else if (a == "-ctk" || a == "-ctv") kv = parse_kv(next());
        else if (a == "-fa") fa = std::string(next()) != "off";
        else if (a == "--warm") n_warm = atoi(next());
        else if (a == "--no-profiler") prof = false;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model.empty()) { fprintf(stderr, "usage: cr_decode_prof -m MODEL [-d DEPTH] [-n TOKENS] [-ctk f16|q8_0] [-ub N] [-ngl N] [-fa on|off]\n"); return 2; }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu;
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { fprintf(stderr, "failed to load %s\n", model.c_str()); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = depth + n_warm + n_tok + 16;
    cp.n_batch = n_ub;
    cp.n_ubatch = n_ub;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = kv;
    cp.type_v = kv;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "failed to create context (n_ctx=%u)\n", cp.n_ctx); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
    unsigned long long rng = 88172645463325252ULL;
    auto rand_tok = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (llama_token) (1000 + rng % (unsigned long long) (n_vocab - 2000)); };

    auto t0 = std::chrono::steady_clock::now();
    std::vector<llama_token> buf(n_ub);
    for (int done = 0; done < depth; ) {
        int n = std::min(n_ub, depth - done);
        for (int i = 0; i < n; i++) buf[i] = rand_tok();
        if (llama_decode(ctx, llama_batch_get_one(buf.data(), n)) != 0) { fprintf(stderr, "depth decode failed at %d\n", done); return 1; }
        done += n;
    }
    llama_synchronize(ctx);
    double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    for (int i = 0; i < n_warm; i++) {
        llama_token t = rand_tok();
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) { fprintf(stderr, "warmup decode failed\n"); return 1; }
        llama_synchronize(ctx);
    }

    if (prof) cudaProfilerStart();
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_tok; i++) {
        llama_token t = rand_tok();
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        llama_synchronize(ctx);
    }
    double dec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    if (prof) cudaProfilerStop();

    printf("{\"depth\": %d, \"n_tok\": %d, \"fill_s\": %.3f, \"decode_s\": %.6f, \"us_per_token\": %.1f}\n",
           depth, n_tok, fill_s, dec_s, dec_s * 1e6 / n_tok);
    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
