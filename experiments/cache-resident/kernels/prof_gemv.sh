# temporary: knob sweep for the gemv kernels
python3 -c "import torch; assert torch.cuda.is_available()" || { echo NO_CUDA; exit 1; }
SH=4096x4096,8192x8192,4096x16384,16384x4096,28672x8192
b() { python3 -m kernels.bench_gemv --shapes $SH --result-exp '' "$@" 2>&1 | grep -E "^ +[0-9]|Error|error|^cr_gemv" | sort -u -k1,1 -s | cat; }
b2() { python3 -m kernels.bench_gemv --shapes $SH --result-exp '' "$@" 2>&1 | grep -E "^ +[0-9]|Error|error|^cr_gemv"; }
t() { python3 -m pytest -q -m gpu tests/test_kernels_gpu.py -k 'gemv and not bench' 2>&1 | tail -1; }
echo TEST; t; echo TEST_PDL; CR_GEMV_PDL=1 t; echo TEST_SREG; CR_GEMV_DBG=256 t; echo TEST_W2; CR_GEMV_W=2 t; echo TEST_W8; CR_GEMV_W=8 t
echo NEW; CR_GEMV_VERBOSE=1 b2
echo NEW_PDL; CR_GEMV_PDL=1 b2
for w in 2 8; do echo "W=$w"; CR_GEMV_W=$w CR_GEMV_VERBOSE=1 b2 --variants lut; done
echo D3; CR_GEMV_D=3 CR_GEMV_VERBOSE=1 b2 --variants lut
echo LOADONLY9; CR_GEMV_DBG=9 b2 --variants lut
