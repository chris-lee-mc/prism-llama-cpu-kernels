# temporary: knob sweep for the gemv kernels
for d in 0 1 2 8 9 11; do echo "DBG=$d"; CR_GEMV_DBG=$d python3 -m kernels.bench_gemv --variants lut --shapes 4096x4096,28672x8192 --result-exp '' 2>&1 | grep -E "^ +[0-9]"; done
for b in 0 8; do echo "BPS=$b"; CR_GEMV_BPS=$b python3 -m kernels.bench_gemv --variants lut --shapes 4096x4096,16384x4096 --result-exp '' 2>&1 | grep -E "^ +[0-9]"; done
echo "BPS=0 DBG=9"; CR_GEMV_BPS=0 CR_GEMV_DBG=9 python3 -m kernels.bench_gemv --variants lut --shapes 4096x4096 --result-exp '' 2>&1 | grep -E "^ +[0-9]"
