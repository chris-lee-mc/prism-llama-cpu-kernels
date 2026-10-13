# temporary: final gemv bench run (copy BW, tests, bench, PDL bench, config overrides)
python3 -c "import torch; assert torch.cuda.is_available()" || { echo NO_CUDA; exit 1; }
L=results/p0_smoke/rtx5090/logs; mkdir -p $L
{
python3 - <<'PY'
import torch
a = torch.empty(1 << 30, dtype=torch.uint8, device="cuda"); b = torch.empty_like(a)
for _ in range(3): b.copy_(a)
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(20): b.copy_(a)
e.record(); e.synchronize()
print(f"COPYBW {2 * 20 * (1 << 30) / (s.elapsed_time(e) * 1e-3) / 1e9:.0f} GB/s (d2d copy, read+write)")
PY
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader
echo TEST; python3 -m pytest -q -m gpu tests/test_kernels_gpu.py -k 'gemv and not bench' 2>&1 | tail -1
echo TEST_PDL; CR_GEMV_PDL=1 python3 -m pytest -q -m gpu tests/test_kernels_gpu.py -k 'gemv and not bench' 2>&1 | tail -1
echo NEW; CR_GEMV_VERBOSE=1 python3 -m kernels.bench_gemv --result-exp '' 2>&1 | grep -E "^ +[0-9]|rror|^cr_gemv|target|device" | uniq
echo NEW_PDL; CR_GEMV_PDL=1 python3 -m kernels.bench_gemv --result-exp '' 2>&1 | grep -E "^ +[0-9]|rror|target"
for o in "CR_GEMV_W=2" "CR_GEMV_W=8" "CR_GEMV_D=3"; do echo "$o"; env $o CR_GEMV_VERBOSE=1 python3 -m kernels.bench_gemv --variants lut --result-exp '' 2>&1 | grep -E "^ +[0-9]|rror|^cr_gemv" | uniq; done
} 2>&1 | tee $L/gemv_final.log
