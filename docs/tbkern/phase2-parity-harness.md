# Phase 2: deterministic parity harness

`tbkern_compare.py` compares the read-only artifacts produced by
`examples/debug/llama-debug`. It does not launch a model, download a GGUF, or
modify either input artifact. Run the same prompt and decode command twice:
once with Prism's default path and once with `GGML_TBKERN_Q2_0=1`, then pass
the resulting files to the comparator.

## Artifact capture

Use identical model, prompt, context, threads, and output directory layout.
For example:

```sh
./build/bin/llama-debug -m model.gguf -p "Hello" -t 8 \
  --save-logits --logits-output-dir results/native
GGML_TBKERN_Q2_0=1 ./build/bin/llama-debug -m model.gguf -p "Hello" -t 8 \
  --save-logits --logits-output-dir results/tbkern
```

The debug utility writes a float32 logits file and a token-id file. Their
names include the model basename; use the exact paths printed by the utility.

## Compare and record metadata

```sh
python3 scripts/tbkern_compare.py \
  --native-logits results/native/llamacpp-model.bin \
  --tbkern-logits results/tbkern/llamacpp-model.bin \
  --native-tokens results/native/llamacpp-model-tokens.bin \
  --tbkern-tokens results/tbkern/llamacpp-model-tokens.bin \
  --json-out results/phase2-parity.json
```

The report contains input byte counts and SHA-256 digests, float count and
finite-value ranges, maximum absolute and relative differences, NMSE, top-1
indices, token equality, and the final gate result. A non-zero exit status
means a gate failed or an artifact was malformed.

The default correctness gates are:

* NMSE <= `1e-4` (the denominator is the sum of squared native logits);
* top-1 index equality for the final logits vector. `llama-debug` writes one
  vector; `--vocab-size`, when supplied, must exactly equal its length and is
  only an explicit shape assertion;
* exact byte equality for tokens when both token files are supplied.

Use `--nmse-tolerance` only when a test explicitly documents a different
tolerance. Both tolerance arguments must be finite and non-negative.
`--relative-floor` affects only the diagnostic max-relative value, not
pass/fail. If the native vector is all zeros, NMSE is considered undefined and
the comparison passes only when the vectors are exactly equal. The JSON schema
identifier is `tbkern.parity.v1` so reports can be archived alongside benchmark
logs.
