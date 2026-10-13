# temporary: knob sweep for the gemv kernels
python3 -c "import torch; assert torch.cuda.is_available()" || { echo NO_CUDA; exit 1; }
SH=4096x4096,28672x8192
b() { python3 -m kernels.bench_gemv --shapes $SH --variants lut --result-exp '' "$@" 2>&1 | grep -E "^ +[0-9]|Error|error"; }
for w in 8 4; do for d in 16 0 32 9 41 73 137 233; do echo "W=$w DBG=$d"; CR_GEMV_W=$w CR_GEMV_DBG=$d b; done; done
