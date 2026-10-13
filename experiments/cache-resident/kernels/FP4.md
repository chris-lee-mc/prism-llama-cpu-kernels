# Variant (b): unpack to FP4 E2M1 in shared memory

Status: implemented as an unpack-and-dot path on CUDA cores. There is no FP4 MMA path. Measured on rtx5090 (see table below).

## What `cr_gemv_fp4` does

Same structure as the LUT variant: weight tiles (4 groups x RB rows) stream through a cp.async ring in shared memory, lane = row,
with H = 2 two warps share a row set (one per 64-weight half of each group). Per group (128 weights) and lane (64 weights, one `uint4`):

1. Convert the 2-bit codes to E2M1 nibbles with SWAR bit ops (`to_e2m1`): `00 -> 0x0`, `01 -> 0x2` (+1.0), `10 -> 0xA` (-1.0), `11 -> 0x0`.
   The result is two words per source word: even weights (0,2,..,14) and odd weights (1,3,..,15). Order inside the tile is a permutation of k, the x pairs are staged to match.
2. Stage the nibbles in a per-warp shared tile (1 KB), `__syncwarp`, read them back.
3. Decode each nibble through a 16-entry shared table (`tab[16]`, full E2M1 value set, ternary uses 3 entries), multiply with x (fp32, staged per tile, all lanes read the same x: broadcast), accumulate in fp32, then `acc += scale * partial`.

The shared round trip is real (write, `__syncwarp`, read) but a lane reads back its own data. On CUDA cores it adds work and gives no benefit; it models the
path an FP4 tensor-core kernel would need (operands staged in shared memory as nibbles).

## Why no MMA

- `mma ... kind::f8f6f4` and block-scaled FP4 MMA on sm_120 take FP4/FP6/FP8 for both operands. Our x is fp16 and the pinned tolerance (`rtol=2e-3`, `atol=2e-3*max|ref|`) does not hold if x is rounded to e4m3 or FP4.
- At batch 1 the MMA would use one of eight columns (or one of sixteen rows). The kernel is bound by weight traffic (4 MB at 4096x4096), not by tensor throughput.
- A mixed path (fp16 MMA with weights decoded to fp16) would still pay the same decode instructions per weight, so it is not an FP4 path.

`CR_FP4_MMA=1` is accepted by `build.py` (adds `-DCR_FP4_MMA` and builds `sm_120a`) but no code uses it yet.

## Expectation (to verify on a GPU)

Per weight the LUT variant needs about 2.5 SM instructions (shift, mask, address, shared load, add per two weights).
The FP4 variant needs about 6 (nibble convert, two table loads per two weights, x load, two FMAs) plus the staging round trip and about 0.4 shared accesses per weight on the shared-memory pipe.
Expect (b) to be issue or shared-memory bound at a lower fraction of DRAM peak than (a). If it is not, the bench will say so.

## Bank-conflict design (checked offline by `tests/test_kernels_emulate.py`)

- `tab[16]`: 16 consecutive words, distinct nibbles hit distinct banks, equal nibbles broadcast.
- x tile: every lane of a warp reads the same address (broadcast).
- Staging stores/loads: lane stride 16 B for each of the two 128-bit accesses.
- Weight tile: 16 B chunk c of row r at c ^ (r & 7), so 8 rows reading the same chunk hit 8 bank groups.

## Measured results

rtx5090 (secure pod, d2d copy 1524 GB/s, driver 580), `python -m kernels.bench_gemv`, cold = rotating pool > 4x L2,
CUDA graph of back-to-back launches. PDL = `CR_GEMV_PDL=1` (weight copies start before the previous kernel ends).

| shape NxK | lut us / %peak | fp4 us / %peak | lut PDL | fp4 PDL |
|---|---|---|---|---|
| 4096x4096 | 6.72 / 37.2 | 8.69 / 28.7 | 4.52 / 55.3 | 5.75 / 43.4 |
| 8192x8192 | 16.94 / 58.8 | 23.43 / 42.5 | 14.13 / 70.5 | 20.89 / 47.7 |
| 4096x16384 | 22.78 / 43.8 | 23.45 / 42.5 | 20.54 / 48.6 | 21.11 / 47.2 |
| 16384x4096 | 14.64 / 68.1 | 21.43 / 46.5 | 12.40 / 80.4 | 18.30 / 54.5 |
| 28672x8192 | 50.32 / 69.3 | 75.28 / 46.3 | 45.33 / 76.9 | 73.38 / 47.5 |

FP4 stays issue bound: about 3.5 instructions and 1.25 shared loads per weight against 1.75 and 0.5 for the LUT.
