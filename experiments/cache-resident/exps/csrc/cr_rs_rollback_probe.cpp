// Rollback probe for recurrent / hybrid models (n_rs_seq > 0): is the state after llama_memory_seq_rm the state of the kept tokens?
// Decode a prompt, then n extra tokens either one at a time (as draft-simple drafts) or as one batch (as a verify batch), roll back
// to m kept tokens, decode one probe token and compare its logits with a context that decoded only the m kept tokens the same way.
// n_extra may be n_rs + 1: a verify batch of K + 1 tokens with n_rs = K. Then kept 1 is the largest rollback the server does (K), kept 0 is out of the window.
// Build: see exps/wrap_b.py build_probe(). Prints one JSON line.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void batch_add(llama_batch & b, llama_token t, llama_pos p, bool logits) {
    b.token[b.n_tokens]     = t;
    b.pos[b.n_tokens]       = p;
    b.n_seq_id[b.n_tokens]  = 1;
    b.seq_id[b.n_tokens][0] = 0;
    b.logits[b.n_tokens]    = logits;
    b.n_tokens++;
}

static bool decode(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos p0, bool single) {
    llama_batch b = llama_batch_init((int) std::max<size_t>(toks.size(), 1), 0, 1);
    bool ok = true;
    for (size_t i = 0; i < toks.size() && ok; ++i) {
        batch_add(b, toks[i], p0 + (llama_pos) i, single || i + 1 == toks.size());
        if (single) {
            ok = llama_decode(ctx, b) == 0;
            b.n_tokens = 0;
        }
    }
    if (!single && ok && b.n_tokens > 0) {
        ok = llama_decode(ctx, b) == 0;
    }
    llama_batch_free(b);
    return ok;
}

struct cmp {
    double max_abs = 0.0;
    double max_ref = 0.0;
    bool   top1_eq = true;
};

static cmp compare(const float * a, const float * b, int n) {
    cmp c;
    int ia = 0, ib = 0;
    for (int i = 0; i < n; ++i) {
        c.max_abs = std::max(c.max_abs, (double) std::fabs(a[i] - b[i]));
        c.max_ref = std::max(c.max_ref, (double) std::fabs(b[i]));
        ia = a[i] > a[ia] ? i : ia;
        ib = b[i] > b[ib] ? i : ib;
    }
    c.top1_eq = ia == ib;
    return c;
}

int main(int argc, char ** argv) {
    std::string model;
    int n_gpu = 0, n_rs = 4, n_prompt = 64, n_extra = 4;
    std::string modes = "both"; // single (one-token decodes), batch (one batch) or both
    unsigned seed = 1234;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "-m") model = next();
        else if (a == "-ngl") n_gpu = atoi(next());
        else if (a == "--n-rs") n_rs = atoi(next());
        else if (a == "--prompt") n_prompt = atoi(next());
        else if (a == "--extra") n_extra = atoi(next());
        else if (a == "--modes") modes = next();
        else if (a == "--seed") seed = (unsigned) atoi(next());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || n_extra < 1 || n_extra > n_rs + 1 || (modes != "both" && modes != "single" && modes != "batch")) {
        fprintf(stderr, "usage: %s -m model.gguf [-ngl N] [--n-rs 4] [--prompt 64] [--extra 4 (<= n-rs + 1)] [--modes both|single|batch] [--seed S]\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu;
    llama_model * mdl = llama_model_load_from_file(model.c_str(), mp);
    if (!mdl) { fprintf(stderr, "failed to load %s\n", model.c_str()); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mdl));

    auto cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_rs_seq = (uint32_t) n_rs;
    llama_context * ctx = llama_init_from_model(mdl, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // fixed pseudo-random token ids (LCG), away from the special tokens at the end of the vocab
    std::vector<llama_token> toks(n_prompt + n_extra + 1);
    unsigned x = seed;
    for (auto & t : toks) { x = x * 1103515245u + 12345u; t = (llama_token) (100 + (x >> 8) % (unsigned) std::max(1, n_vocab - 1000)); }
    const std::vector<llama_token> prompt(toks.begin(), toks.begin() + n_prompt);
    const std::vector<llama_token> extra(toks.begin() + n_prompt, toks.begin() + n_prompt + n_extra);
    const llama_token probe = toks.back();
    auto * mem = llama_get_memory(ctx);

    // logits of the probe token after: prompt, then extra[0, n_dec) decoded single or batched, then a rollback to m kept tokens (n_dec == m: none)
    auto run = [&](int n_dec, int m, bool single, bool & rm_ok) -> std::vector<float> {
        llama_memory_clear(mem, true);
        std::vector<float> out;
        rm_ok = true;
        if (!decode(ctx, prompt, 0, false)) return out;
        if (n_dec > 0 && !decode(ctx, std::vector<llama_token>(extra.begin(), extra.begin() + n_dec), n_prompt, single)) return out;
        if (n_dec > m) rm_ok = llama_memory_seq_rm(mem, 0, n_prompt + m, -1);
        if (!decode(ctx, { probe }, n_prompt + m, true)) return out;
        const float * l = llama_get_logits_ith(ctx, -1);
        out.assign(l, l + n_vocab);
        return out;
    };

    printf("{\"model\": \"%s\", \"n_vocab\": %d, \"n_rs_seq\": %u, \"n_prompt\": %d, \"n_extra\": %d, \"ngl\": %d, \"cases\": [",
           model.substr(model.find_last_of('/') + 1).c_str(), n_vocab, llama_n_rs_seq(ctx), n_prompt, n_extra, n_gpu);
    bool first = true;
    for (int m = 0; m < n_extra; ++m) {
        for (int single = 1; single >= 0; --single) {
            if ((single && modes == "batch") || (!single && modes == "single")) continue;
            if (single && n_extra - m > n_rs) continue; // one-token decodes: draft-simple never rolls back more than n_rs
            bool ok_ref = true, ok_rm = true;
            const auto ref = run(m, m, single, ok_ref);
            const auto got = run(n_extra, m, single, ok_rm);
            if (ref.empty() || got.empty()) {
                printf("%s{\"kept\": %d, \"rollback\": %d, \"decode\": \"%s\", \"error\": \"decode failed\"}", first ? "" : ", ", m, n_extra - m, single ? "single" : "batch");
            } else {
                const cmp c = compare(got.data(), ref.data(), n_vocab);
                printf("%s{\"kept\": %d, \"rollback\": %d, \"decode\": \"%s\", \"seq_rm_ok\": %s, \"max_abs_diff\": %.6g, \"max_abs_ref\": %.6g, \"top1_equal\": %s}",
                       first ? "" : ", ", m, n_extra - m, single ? "single" : "batch", ok_rm ? "true" : "false", c.max_abs, c.max_ref, c.top1_eq ? "true" : "false");
            }
            first = false;
        }
    }
    printf("]}\n");

    llama_free(ctx);
    llama_model_free(mdl);
    llama_backend_free();
    return 0;
}
