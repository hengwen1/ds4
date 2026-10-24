# Jetson Thor

[README](../README.md) | [Getting started](../README.md#start-here)

Jetson Thor is a Blackwell-generation module with the same 128 GB unified
memory pool as the DGX Spark, so the model recipes carry over unchanged. The
**build does not**: Thor is compute capability 11.0 (`sm_110a`) and GB10 is
12.1 (`sm_121a`), and the two Blackwell variants do not expose the same
instructions. In particular the block-scaled MXFP4 `mma.sync` form used by the
Spark MXFP4 kernels does not exist on `sm_110a`; there the equivalent product
comes from the fifth-generation tensor core (tcgen05).

## Build and run

Use the CUDA toolkit that ships with the JetPack release for the module; the
build needs `nvcc` and cuBLAS, and `sm_110a` targets (with tcgen05) require
CUDA 13 or newer. Check that `nvidia-smi` sees the GPU before building.

```sh
make cuda-thor
./download_model.sh ds4f-q2
./ds4 --cuda
```

`make cuda-thor` compiles with `-gencode arch=compute_110a,code=sm_110a` and
`-DDS4_CUDA_HAVE_TCGEN05=1`. The binary is architecture-specific: a
`make cuda-spark` build does not run on Thor and a Thor build does not run on
GB10 (the shipped flags carry no PTX fallback). The startup line names the
backend it was built for:

```
ds4: CUDA backend initialized on NVIDIA Thor (sm_110) dev=0
ds4: CUDA platform: Jetson Thor, tcgen05 kernels compiled in
```

Keep other inference services stopped before loading a model; on a unified
memory module they compete with the weights directly. Restore them when
finished. `make cuda-thor` needs roughly the same build time and disk as
`make cuda-spark`.

## What the Thor build enables

**tcgen05 MXFP4 light-indexer scorer.** The DeepSeek light indexer scores every
(verification token, compressed component) pair against 64 indexer heads, and
on Spark that runs as per-head `mma.sync` MXFP4 block-scaled products. On Thor
the same E2M1 x E2M1 block-scaled product is issued as
`tcgen05.mma.kind::mxf4.block_scale.scale_vec::2X`: 128 components of the tile
on the M side, the 8 heads x 16 tokens of a head group on the N side, two
accumulating K=64 instructions covering the 128-wide head dimension. Because
the (token, head) pairs land on the N side, the head reduction is a register
sum in the epilogue. The path keeps the encoder, the causal mask and the
double-precision-checked numerics of the Spark scorer, and stays behind the
same guards (quality mode, `DS4_CUDA_NO_INDEXER_MXF4=1`, shape limits, a
non-capturing stream, one device).

Measured on one Thor, driver 595.78 / CUDA 13.2, with another service holding
about 96 GB of the pool, best of eight alternating runs of the indexer call
(`tests/test_cuda_indexer_mxfp4 --bench`, `pos0` at the end of the context so
every component is visible, i.e. the shape a prefill or verification batch
produces):

| Tokens | Components | fp16 WMMA scorer | tcgen05 MXFP4 | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 4096 | 0.186 ms | 0.051 ms | 3.6x |
| 128 | 4096 | 0.506 ms | 0.128 ms | 3.9x |
| 256 | 8192 | 1.80 ms | 0.408 ms | 4.4x |
| 512 | 8192 | 3.49 ms | 0.756 ms | 4.6x |
| 1024 | 16384 | 13.35 ms | 2.75 ms | 4.9x |
| 2048 | 16384 | 26.3 ms | 5.43 ms | 4.8x |

The MXFP4 column includes the per-call encoder for the query rows and key
components; the WMMA column needs no encode. These are indexer-kernel timings,
not whole-model throughput. Clocks ramp on this module, so the first
measurement after an idle period is slower; interleave the two arms and repeat.

**Everything else** runs the generic Blackwell CUDA paths. The GB10-tuned
selection in `ds4_cuda.cu` and the vendored MMQ tier are keyed on
`ds4_gpu_device_is_spark()`, i.e. capability 12.1, and stay off on Thor; those
kernels were measured on Spark and have no Thor numbers yet. Expect
Spark-class behavior rather than identical timings. If you are testing a
Spark-specific kernel on Thor, `DS4_CUDA_PLATFORM=spark` forces the
classification for that build; the tcgen05 dispatch still requires the real
capability, so a forced platform cannot launch a kernel the device lacks.

## DeepSeek V4.1 Flash and larger weights

The Spark instructions apply unchanged: Q2 needs SSD streaming on a single
128 GB module, and Q4 does not fit.

```sh
./download_model.sh ds41f-q2
./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming --ctx 32768
```

## GLM 5.3 Flash Q4

The 178 GiB Q4 file also runs here, through the same type routing as on Spark:
the Q4_K routed experts take the generic routed MoE (`12/12/12` expert layers),
not the CUDA GLM kernels, which are Q2_K-only. Size the expert cache explicitly;
the automatic one is about 1 GiB on a 128 GB machine and leaves the run I/O
bound.

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q4_K.gguf --ssd-streaming \
  --ssd-streaming-cache-experts 32GB --ctx 65536
```

Verified on one Thor (122 GB pool, 191 GB file on local NVMe) with a 24 GiB
expert cache: 1.6 t/s prefill and 1.4-1.9 t/s decode at 8K context, and correct,
coherent answers. Those numbers measure the SSD hit rate, not the GPU: GLM is
disk-bound at this cache size on a module that cannot hold the file.

## Testing

Model-free checks that cover the Thor-specific code:

```sh
make test-cuda-indexer-mxfp4     # 20 shapes vs a double-precision quantized oracle
./tests/test_cuda_indexer_mxfp4 --bench
make test-cuda-indexer           # full indexer regression, all dispatch arms
```

The MXFP4 test disables the other indexer arms, so a dispatch miss fails
instead of silently timing the fp16 scorer. Run both under Compute Sanitizer
when changing the tcgen05 staging, descriptors or epilogue.
