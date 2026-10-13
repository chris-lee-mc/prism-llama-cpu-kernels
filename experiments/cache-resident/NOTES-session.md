# Session constraints (orchestrator notes)

- Branch: exp/cache-resident, based on upstream PrismML-Eng/llama.cpp `prism`. Fork-only commits from chris-lee-mc `prism` are NOT carried; port only if an experiment needs them.
- No local GPU. All GPU work runs on RunPod pods via RUNPOD_API_KEY (GraphQL https://api.runpod.io/graphql, REST https://rest.runpod.io/v1).
- Local compile-check toolchain: /usr/local/cuda-12.8/bin/nvcc (can target sm_120/sm_89/sm_86 offline). Python venv: /home/user/venv (torch CPU 2.14, cuda-python 12.8, numpy, pytest, requests, matplotlib).
- Repo is public: pods can `git clone -b exp/cache-resident https://github.com/chris-lee-mc/prism-llama-cpu-kernels`.
- Pods have NO GitHub credentials. Results path (replaces P0.6 "push from pod"): pod writes results/<exp>/<gpu>/<timestamp>.json and serves the results dir over HTTP on port 8000; orchestrator fetches via https://<podId>-8000.proxy.runpod.net/ and commits/pushes from the container.
- Every pod must have: budget gate (refuse launch if clientBalance < est cost + $2 margin or session spend cap exceeded), 40-min failsafe self-terminate (pod-side `runpodctl remove pod $RUNPOD_POD_ID` or sleep-then-exit), and orchestrator-side termination in a finally block.
- AGENTS.md: ASCII only in code/comments (no em dash, arrows, x-sign), concise comments, commits use `Assisted-by: Claude` (never Co-authored-by).
- Schedule is in days, not weeks. No card purchase decision. RunPod is primary for every GPU incl. PRO 6000 (cheaper than GCP); a GCP PRO 6000 is fallback only, so the pod-side runner should also work as a plain local run on any CUDA host.
- RunPod account auto-tops-up when balance < $25: the budget gate must NOT refuse launches just because clientBalance is low; gate on a per-session spend cap (ledger) and per-run estimated cost instead.
- End goal: PR to PrismML llama.cpp fork (see PLAN.md section 10). All judge stages must include the upstreamability check defined there.
