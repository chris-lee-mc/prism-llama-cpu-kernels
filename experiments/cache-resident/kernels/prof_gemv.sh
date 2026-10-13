# temporary: knob sweep for the gemv kernels
python3 -c "import torch; assert torch.cuda.is_available()" || { echo NO_CUDA; exit 1; }
SH=4096x4096,8192x8192,16384x4096,28672x8192
b() { python3 -m kernels.bench_gemv --shapes $SH --result-exp '' "$@" 2>&1 | grep -E "^ +[0-9]|gap|Error|error"; }
t() { python3 -m pytest -q -m gpu tests/test_kernels_gpu.py -k 'gemv and not bench' 2>&1 | tail -1; }
echo TEST; t; echo TEST_W8; CR_GEMV_W=8 t; echo TEST_PDL; CR_GEMV_PDL=1 t
echo NEW; b
for w in 4 8; do for p in 2 4 6; do echo "W=$w BPS=$p"; CR_GEMV_W=$w CR_GEMV_BPS=$p b --variants lut; done; done
echo NEW_PDL; CR_GEMV_PDL=1 b
echo FLOOR16; CR_GEMV_DBG=16 b --variants lut
echo LOADONLY9; CR_GEMV_DBG=9 b --variants lut
