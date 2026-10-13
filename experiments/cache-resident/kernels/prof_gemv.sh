# temporary: knob sweep for the gemv kernels
python3 -c "import torch; assert torch.cuda.is_available()" || { echo NO_CUDA; exit 1; }
SH=4096x4096,8192x8192,28672x8192
b() { python3 -m kernels.bench_gemv --shapes $SH --result-exp '' "$@" 2>&1 | grep -E "^ +[0-9]|gap"; }
echo NEW; b
echo NEW_PDL; CR_GEMV_PDL=1 b
echo FLOOR16; CR_GEMV_DBG=16 b --variants lut
echo FLOOR16_BPS8; CR_GEMV_DBG=16 CR_GEMV_BPS=8 b --variants lut
F=kernels/csrc/ternary_gemv.cu; cp $F /tmp/new.cu
git fetch -q --depth 1 origin exp/cache-resident && git show FETCH_HEAD:experiments/cache-resident/$F > $F && echo OLD && b
cp /tmp/new.cu $F
