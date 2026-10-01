# AVX-VNNI INT8 演示与技能包

围绕 **AVX-VNNI**（VEX 编码的 `vpdpbusd` 家族）的一组可运行示例与配套 Skill。
VNNI 把 INT8 推理里最常见的「乘-加-累加」三步
（`vpmaddubsw` + `vpmaddwd` + `vpaddd`）压缩成一条指令，
并且**没有 16-bit 中间结果饱和问题**。

```
VPDPBUSD ymm1, ymm2, ymm3/m256         ; AVX-VNNI,   256-bit, VEX 编码
VPDPBUSD zmm1{k1}{z}, zmm2, zmm3/m512  ; AVX512-VNNI, 512-bit, EVEX 编码

for j in 0..7:                          ; 8 个 int32 lane
    for k in 0..3:                      ; 每个 lane 吃 4 个【连续】字节
        tmp += (uint8_t)a[4*j+k] * (int8_t)b[4*j+k];
        ;      ^^^^^^ 无符号         ^^^^^ 有符号（非对称，不是笔误）
    dst[j] = src[j] + tmp;              ; 回绕(W) 或 有符号饱和(S)
```

## 目录结构

```
.
├── avx-vnni.zip              # 打包好的 Skill（SKILL.md + scripts/ + references/ + assets/）
├── README.md
└── src/
    ├── vnni_demo.c           # 主演示：CPUID 检测 / 语义 / 点积 / 量化 GEMM / 计时
    ├── vnni_asm.asm          # 手写 MASM64 点积，手工 emit VEX 机器码
    ├── build.bat             # 一键编译（cl + ml64 + link），并直接运行
    └── sample_output.txt     # 在「有 AVX-VNNI、无 AVX-512」CPU 上的实测输出
```

`avx-vnni.zip` 是可直接安装的 Skill 包，解压后结构：

```
avx-vnni/
├── SKILL.md                       # 技能说明：何时使用、快速参考、工作流、7 条坑
├── assets/                        # 与 src/ 中的 demo 同源
│   ├── vnni_demo.c  vnni_asm.asm  build_msvc.bat  sample_output.txt
├── scripts/
│   ├── vex_vnni_encode.py         # 计算 10 条 VNNI 指令的 VEX.256 字节（与 dumpbin 逐字节一致）
│   └── check_vnni_encoding.py     # 用 dumpbin 检查 .obj/.exe 中是否混入 EVEX 编码
└── references/
    ├── intrinsics-and-encoding.md  # 各编译器 intrinsic 对照、CPUID、VEX 位域
    ├── kernel-patterns.md          # 点积 / GEMM / 横向归约模式与数据布局
    └── msvc-toolchain.md           # 定位 cl.exe/ml64.exe、/utf-8、dumpbin、x64 ABI
```

## 编译与运行

Windows + MSVC（需安装 MASM64 组件）：

```bat
cd src
build.bat
```

或手动三步：

```bat
cl   /nologo /O2 /arch:AVX2 /utf-8 /std:c17 /EHsc /W3 /c vnni_demo.c
ml64 /nologo /c /Cx /W3 vnni_asm.asm
link /nologo /subsystem:console /machine:x64 vnni_demo.obj vnni_asm.obj /out:vnni_demo.exe
vnni_demo.exe
```

GCC / Clang（此时 `dot_vnni_asm` 未定义，需自行去掉该调用或改用 GAS 版本）：

```sh
gcc -O3 -mavx2 -mavxvnni vnni_demo.c -o vnni_demo
```

> `build.bat` 里的 `VSDIR` 是写死的路径，换机器需要改。

## CPU 支持与检测

| 特性 | CPUID | 位 |
|---|---|---|
| AVX2（前置条件） | leaf `07H`, ECX=0 | EBX[5] |
| AVX512-VNNI | leaf `07H`, ECX=0 | ECX[11] |
| **AVX-VNNI** | leaf `07H`, ECX=**1** | **EAX[4]** |
| AVX-VNNI_INT16 | leaf `07H`, ECX=1 | EAX[10] |

还要用 `XGETBV` 确认 OS 已开启对应状态：
YMM 需要 `XCR0 & 0x6 == 0x6`，ZMM 需要 `XCR0 & 0xE6 == 0xE6`。

支持 AVX-VNNI 的 CPU：Intel Alder Lake+ / Sapphire Rapids+、AMD Zen4+。
**Arrow Lake / Lunar Lake 有 AVX-VNNI 但没有 AVX-512。** 必须运行时检测，不要假设。

## 实测性能

数据取自 [sample_output.txt](src/sample_output.txt)（Alder Lake 级别、无 AVX-512 的机器，
N=16384，取 3 次最优）：

| 实现 | 耗时 | 吞吐 |
|---|---|---|
| 标量 | 12.63 ms | 25.95 GMAC/s / 51.90 GB/s |
| 纯 AVX2 模拟 | 3.74 ms | 87.71 GMAC/s / 175.43 GB/s |
| **AVX-VNNI (intrinsic)** | **1.87 ms** | **175.11 GMAC/s / 350.22 GB/s** |
| AVX-VNNI (手写 MASM) | 1.88 ms | 173.84 GMAC/s / 347.67 GB/s |

量化 GEMM `C[96][96] = A[96][256](u8) × B[96][256](s8)`：标量 47 GMAC/s → VNNI 319 GMAC/s，
结果全部一致。

瓶颈是 **load 端口（2 × 32B load/cycle）而非算力**，继续加深展开不会再有收益。

## 两个必须知道的坑

### 1. MSVC 会把 `_mm256_dpbusd_epi32` 编成 EVEX

不带 `_avx` 后缀的两个 u8×s8 intrinsic 会生成 `62 ...`（AVX512-VNNI），
在 Alder Lake / Arrow Lake / Zen4 上运行即 `#UD`（进程退出码 132）。
必须改用带 `_avx` 后缀的版本生成 `C4 ...`（VEX）：

```c
#if defined(_MSC_VER)
#  define DPBUSD_256(c,a,b)  _mm256_dpbusd_avx_epi32 ((c),(a),(b))
#  define DPBUSDS_256(c,a,b) _mm256_dpbusds_avx_epi32((c),(a),(b))
#else
#  define DPBUSD_256(c,a,b)  _mm256_dpbusd_epi32     ((c),(a),(b))
#  define DPBUSDS_256(c,a,b) _mm256_dpbusds_epi32    ((c),(a),(b))
#endif
```

`ml64` 同样对 `vpdpbusd` 输出 EVEX，且不支持 `{vex3}`/`{vex2}` 限定符
（A2008/A2009），因此 [vnni_asm.asm](src/vnni_asm.asm) 里用 `db` 手工 emit VEX 字节。

### 2. Windows x64 ABI 下 YMM6–YMM15 是非易失寄存器

手写汇编若使用 `ymm6`/`ymm7` 却不保存，会静默破坏调用方状态。
本 demo 曾因此测出 `3.7e-289` 秒的垃圾计时值，100% 稳定复现、极难排查。
手写汇编请只用 YMM0–YMM5（可复用两个作临时寄存器），或在栈上保存/恢复。

## 参考

- Intel SDM Vol.2 Table 2-25（VEX 前缀位域）、Vol.2 `VPDPBUSD` 指令说明
- Skill 包 `SKILL.md` 的 Pitfalls 一节（共 7 条，含数据布局、横向归约开销、
  `/utf-8` 编码、纯 AVX2 模拟在大数值下会算错等）
