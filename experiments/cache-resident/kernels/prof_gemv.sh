L=results/p0_smoke/rtx5090/logs; mkdir -p $L
python3 -m pytest -q -m gpu tests/test_kernels_gpu.py -k 'gemv and not bench' 2>&1 | tail -3
python3 -m kernels.bench_gemv --result-exp p0_gemv
CR_GEMV_PDL=1 python3 -m kernels.bench_gemv --result-exp p0_gemv_pdl
P="import torch; from kernels import ternary as T
for N,K in ((4096,4096),(28672,8192)):
    q,s,_=T.random_ternary(N,K,device='cuda'); x=torch.randn(K,device='cuda').half()
    for i in range(3): T.gemv_lut(q,s,x); T.gemv_fp4(q,s,x)
torch.cuda.synchronize()"
/usr/local/cuda/bin/ncu --set full -k regex:cr_gemv --launch-skip 4 --launch-count 2 python3 -c "$P" > $L/ncu_4096.txt 2>&1
/usr/local/cuda/bin/ncu --set full -k regex:cr_gemv --launch-skip 10 --launch-count 2 python3 -c "$P" > $L/ncu_28672.txt 2>&1
SO=$(find / -name 'cr_ternary_ext*.so' 2>/dev/null | head -1); echo so=$SO
/usr/local/cuda/bin/cuobjdump -sass "$SO" > $L/sass.txt 2>&1; wc -l $L/*.txt
