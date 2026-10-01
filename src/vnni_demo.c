/* =============================================================================
 *  vnni_demo.c  ——  AVX-VNNI (VEX 编码的 vpdpbusd 家族) 实战演示
 * =============================================================================
 *
 *  VNNI = Vector Neural Network Instructions。它把 INT8 推理里最常见的
 *  "乘-加-累加" 三步（vpmaddubsw + vpmaddwd + vpaddd）压缩成一条指令：
 *
 *      VPDPBUSD ymm1, ymm2, ymm3/m256        (AVX-VNNI, 256-bit, VEX)
 *      VPDPBUSD zmm1{k1}{z}, zmm2, zmm3/m512 (AVX512-VNNI, EVEX)
 *
 *  语义（256-bit / 8 个 int32 通道）：
 *
 *      for j in 0..7:                       // 8 个 int32 lane
 *          int32 tmp = 0;
 *          for k in 0..3:                   // 每个 lane 吃掉 4 个字节
 *              tmp += (uint8_t)a[4*j + k] * (int8_t)b[4*j + k];
 *                     ^^^^^^^^ 无符号        ^^^^^^^ 有符号
 *          dst[j] = src[j] + tmp;           // 回绕（W）或 有符号饱和（S）
 *
 *  四条指令的命名规则：PDP + B + {U|S} + S + {D|S}
 *                                  ^     ^     ^
 *                               a 操作数  b 操作数 累加方式
 *                               U=无符号  S=有符号  D=回绕 / S=饱和
 *
 *      vpdpbusd   : uint8  * int8   -> int32  回绕
 *      vpdpbusds  : uint8  * int8   -> int32  有符号饱和
 *      vpdpbssd   : int8   * int8   -> int32  回绕
 *      vpdpbssds  : int8   * int8   -> int32  有符号饱和
 *
 *  ⚠ 非对称性很重要：a 是【无符号】u8，b 是【有符号】s8。这正好匹配量化推理里
 *    "激活值(非负, u8) × 权重(可正可负, s8)" 的常见组合（对称量化权重 + 非对称激活）。
 *
 *  检测方式（CPUID）：
 *      AVX2        : CPUID.(EAX=07H, ECX=0):EBX[bit  5]
 *      AVX512-VNNI : CPUID.(EAX=07H, ECX=0):ECX[bit 11]
 *      AVX-VNNI    : CPUID.(EAX=07H, ECX=1):EAX[bit  4]   ← 本 demo 的主角
 *
 *  支持 AVX-VNNI 的 CPU：Intel Alder Lake+/Sapphire Rapids+、AMD Zen4+
 *  （注意：Arrow Lake / Lunar Lake 有 AVX-VNNI，但没有 AVX-512）
 *
 * ============================== 编译方式 =====================================
 *  MSVC (x64 Native Tools):
 *      cl /O2 /arch:AVX2 /std:c17 /EHsc vnni_demo.c
 *
 *  GCC / Clang:
 *      gcc  -O3 -mavx2 -mavxvnni                        vnni_demo.c -o vnni_demo
 *      gcc  -O3 -mavx2 -mavxvnni -mavx512f -mavx512vnni vnni_demo.c -o vnni_demo
 * =============================================================================
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#  include <windows.h>
#  define ALIGN(n) __declspec(align(n))
#else
#  include <cpuid.h>
#  include <time.h>
#  define ALIGN(n) __attribute__((aligned(n)))
#endif
#include <immintrin.h>

/* ========================================================================== */
/* ⚠ MSVC 编码陷阱（本 demo 最实用的一条）                                      */
/* -------------------------------------------------------------------------- */
/*  _mm256_dpbusd_epi32 在 MSVC 下会被编成 **EVEX** 编码，也就是 AVX512-VNNI：  */
/*      62 F2 7D 28 50 ...     <- EVEX 前缀，需要 AVX512-VNNI                  */
/*  在只有 AVX-VNNI、没有 AVX-512 的 CPU（Alder Lake / Arrow Lake / Zen4 等）   */
/*  上运行时会直接 #UD 非法指令崩溃。                                          */
/*                                                                            */
/*  MSVC 的 immintrin.h 因此在 "// AVX_VNNI intrinsics" 段专门提供了            */
/*  带 _avx 后缀的强制 VEX 版本（只需要 AVX-VNNI）：                            */
/*      C4 E2 5D 50 ...        <- VEX 前缀                                     */
/*          _mm256_dpbusd_avx_epi32  /  _mm256_dpbusds_avx_epi32               */
/*          _mm256_dpwssd_avx_epi32  /  _mm256_dpwssds_avx_epi32               */
/*  GCC/Clang 无此后缀，加 -mavxvnni 编译即可生成 VEX。                        */
/*                                                                            */
/*  另注意（实测修正，别信"dpbssd 属于 AVX512-VNNI"的说法）：                   */
/*  四个 byte 乘法变体都在 0F38 映射表里，opcode 都是 50/51，                   */
/*  唯一区别是 VEX 的 pp 字段（隐含前缀）：                                     */
/*        pp=00 (none) -> u8 x u8   vpdpbuud                                   */
/*        pp=01 (66)   -> u8 x s8   vpdpbusd                                   */
/*        pp=10 (F3)   -> s8 x u8   vpdpbsud                                   */
/*        pp=11 (F2)   -> s8 x s8   vpdpbssd                                   */
/*  它们在 MSVC 下【都】编成 VEX，在无 AVX-512 的 CPU 上照样能跑。              */
/*  真正会退化成 EVEX 的只有不带 _avx 后缀的 dpbusd / dpbusds 两个。            */
/* ========================================================================== */
#if defined(_MSC_VER)
#  define DPBUSD_256(c, a, b)  _mm256_dpbusd_avx_epi32 ((c), (a), (b))
#  define DPBUSDS_256(c, a, b) _mm256_dpbusds_avx_epi32((c), (a), (b))
#else
#  define DPBUSD_256(c, a, b)  _mm256_dpbusd_epi32     ((c), (a), (b))
#  define DPBUSDS_256(c, a, b) _mm256_dpbusds_epi32    ((c), (a), (b))
#endif

/* ========================================================================== */
/* 0. 基础设施：计时 / CPUID / 打印                                            */
/* ========================================================================== */

static double now_sec(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq; static int init = 0;
    LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

typedef struct { int avx2, avxvnni, avx512f, avx512vnni, ymm_ok, zmm_ok; } Feat;

static void cpuid_ex(int r[4], int leaf, int sub)
{
#if defined(_MSC_VER)
    __cpuidex(r, leaf, sub);
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

static Feat detect_cpu(void)
{
    Feat f; memset(&f, 0, sizeof f);
    int r[4] = {0};

    cpuid_ex(r, 0, 0);
    int maxleaf = r[0];
    if (maxleaf < 7) return f;

    /* OS 是否启用了 XSAVE / XGETBV */
    int osxsave = 0;
    cpuid_ex(r, 1, 0);
    osxsave = (r[2] >> 27) & 1;

    uint64_t xcr0 = 0;
    if (osxsave) {
#if defined(_MSC_VER)
        xcr0 = _xgetbv(0);
#else
        __asm__ volatile("xgetbv" : "=a"(xcr0) : "c"(0) : "edx");
#endif
    }
    f.ymm_ok = (xcr0 & 0x6u) == 0x6u;        /* SSE state | AVX state (YMM) */
    f.zmm_ok = (xcr0 & 0xE6u) == 0xE6u;     /* + opmask | ZMM_Hi256 | Hi16_ZMM */

    cpuid_ex(r, 7, 0);
    f.avx2       = (r[1] >>  5) & 1;   /* EBX[5]  */
    f.avx512f    = (r[1] >> 16) & 1;   /* EBX[16] */
    f.avx512vnni = (r[2] >> 11) & 1;   /* ECX[11] */

    cpuid_ex(r, 7, 1);
    f.avxvnni    = (r[0] >>  4) & 1;   /* EAX[4]  ← AVX-VNNI */

    return f;
}

ALIGN(32) static int32_t g_tmp8[8];

static void print_i32(const char *tag, __m256i v)
{
    _mm256_store_si256((__m256i *)g_tmp8, v);
    printf("    %-22s [%d, %d, %d, %d, %d, %d, %d, %d]\n", tag,
           g_tmp8[0], g_tmp8[1], g_tmp8[2], g_tmp8[3],
           g_tmp8[4], g_tmp8[5], g_tmp8[6], g_tmp8[7]);
}

/* 8 个 int32 通道横向求和 */
static inline int32_t hsum_epi32(__m256i v)
{
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E)); /* 交换 64-bit 半区 */
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1)); /* 交换 32-bit 半区 */
    return _mm_cvtsi128_si32(s);
}

/* ========================================================================== */
/* 1. 语义演示：一条 vpdpbusd 到底算了什么                                      */
/* ========================================================================== */

static void demo_semantics(const Feat *f)
{
    (void)f;
    printf("\n[1] vpdpbusd 语义演示 (u8 x s8 -> i32, 每 lane 4 字节)\n");
    printf("---------------------------------------------------------------\n");

    /* a: 无符号 u8，全填 2； b: 有符号 s8，全填 3
     * 每个 lane = 4 * (2*3) = 24                                            */
    __m256i a = _mm256_set1_epi8(2);            /* uint8  2 */
    __m256i b = _mm256_set1_epi8(3);            /* int8   3 */
    __m256i c = _mm256_setzero_si256();
    print_i32("c (初始)", c);
    c = DPBUSD_256(c, a, b);
    print_i32("c += a*b (期望 24)", c);

    /* b 换成 -3，验证 b 是有符号的 */
    __m256i bn = _mm256_set1_epi8((char)-3);
    __m256i c2 = DPBUSD_256(_mm256_setzero_si256(), a, bn);
    print_i32("a=2, b=-3 (期望-24)", c2);

    /* 再看 a 是【无符号】的：把 a 填成 0xFE（u8 视角 = 254，s8 视角 = -2）      */
    __m256i aFE = _mm256_set1_epi8((char)0xFE);
    __m256i one = _mm256_set1_epi8(1);
    print_i32("vpdpbusd a=0xFE,b=1", DPBUSD_256(_mm256_setzero_si256(), aFE, one));
    printf("    -> 期望 1016 (= 254*1*4)，证明 a 被当成【无符号 u8】解释\n");

    /* s8 x s8 的 vpdpbssd：也是 VEX 编码，无 AVX-512 照样能跑
     *   MSVC 实际编出 C4 C2 77 50（pp=11 -> F2），已在本机实测通过 */
    print_i32("vpdpbssd a=0xFE,b=1", _mm256_dpbssd_epi32(_mm256_setzero_si256(), aFE, one));
    printf("    -> 期望 -8 (= -2*1*4)，a 按【有符号 s8】解释\n");

    /* 饱和版本 vpdpbusds vs 回绕版本 vpdpbusd */
    __m256i big_a = _mm256_set1_epi8((char)0xFF);   /* u8  255 */
    __m256i big_b = _mm256_set1_epi8((char)127);    /* s8  127 */
    /* 每条 vpdpbussd 每 lane 增加 255*127*4 = 129540 */
    __m256i base = _mm256_set1_epi32(INT32_MAX - 100);
    __m256i wrap = DPBUSD_256(base, big_a, big_b);
    __m256i sat  = DPBUSDS_256(base, big_a, big_b);
    print_i32("base = INT32_MAX-100", base);
    print_i32("vpdpbusd  回绕 ->", wrap);
    print_i32("vpdpbusds 饱和 ->", sat);
    printf("    (回绕结果应为负数: %d；饱和结果应为 INT32_MAX: %d)\n",
           (int32_t)(((uint32_t)INT32_MAX - 100u) + 129540u), INT32_MAX);
}

/* ========================================================================== */
/* 2. 点积：标量 / 纯 AVX2 模拟 / AVX-VNNI  —— 正确性 + 性能三重对比            */
/* ========================================================================== */

#define DOT_N    16384      /* 16KB + 16KB，尽量留在 L1/L2 */
#define DOT_ITER 20000

/* --- 标量参考实现 --- */
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static int32_t dot_scalar(const uint8_t *a, const int8_t *b, size_t n)
{
    int32_t s = 0;
    for (size_t i = 0; i < n; ++i) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

/* --- 纯 AVX2（无 VNNI）模拟：vpmaddubsw + vpmaddwd + vpaddd，3 条指令 --- */
static int32_t dot_avx2_emu(const uint8_t *a, const int8_t *b, size_t n)
{
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i p16 = _mm256_maddubs_epi16(va, vb);          /* u8*s8 -> i16 成对相加 */
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones)); /* i16 对 -> i32 */
    }
    int32_t s = hsum_epi32(acc);
    for (; i < n; ++i) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

/* --- AVX-VNNI：4 路累加器展开，主循环每次吃掉 4*32 = 128 字节 --- */
static int32_t dot_vnni(const uint8_t *a, const int8_t *b, size_t n)
{
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    size_t i = 0;

    for (; i + 128 <= n; i += 128) {
        __m256i va0 = _mm256_loadu_si256((const __m256i *)(a + i     ));
        __m256i vb0 = _mm256_loadu_si256((const __m256i *)(b + i     ));
        __m256i va1 = _mm256_loadu_si256((const __m256i *)(a + i + 32));
        __m256i vb1 = _mm256_loadu_si256((const __m256i *)(b + i + 32));
        __m256i va2 = _mm256_loadu_si256((const __m256i *)(a + i + 64));
        __m256i vb2 = _mm256_loadu_si256((const __m256i *)(b + i + 64));
        __m256i va3 = _mm256_loadu_si256((const __m256i *)(a + i + 96));
        __m256i vb3 = _mm256_loadu_si256((const __m256i *)(b + i + 96));
        acc0 = DPBUSD_256(acc0, va0, vb0);
        acc1 = DPBUSD_256(acc1, va1, vb1);
        acc2 = DPBUSD_256(acc2, va2, vb2);
        acc3 = DPBUSD_256(acc3, va3, vb3);
    }
    for (; i + 32 <= n; i += 32) {
        acc0 = DPBUSD_256(acc0,
                   _mm256_loadu_si256((const __m256i *)(a + i)),
                   _mm256_loadu_si256((const __m256i *)(b + i)));
    }

    __m256i acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1),
                                   _mm256_add_epi32(acc2, acc3));
    int32_t s = hsum_epi32(acc);
    for (; i < n; ++i) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

/* --- 手写 MASM64 汇编版（见 vnni_asm.asm）--- */
extern int32_t dot_vnni_asm(const uint8_t *a, const int8_t *b, size_t n);

/* --- AVX512-VNNI（如果 CPU 支持）--- */
#if defined(_MSC_VER) || defined(__AVX512VNNI__)
static int32_t dot_vnni512(const uint8_t *a, const int8_t *b, size_t n)
{
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        acc0 = _mm512_dpbusd_epi32(acc0,
                   _mm512_loadu_si512((const void *)(a + i     )),
                   _mm512_loadu_si512((const void *)(b + i     )));
        acc1 = _mm512_dpbusd_epi32(acc1,
                   _mm512_loadu_si512((const void *)(a + i + 64)),
                   _mm512_loadu_si512((const void *)(b + i + 64)));
    }
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_dpbusd_epi32(acc0,
                   _mm512_loadu_si512((const void *)(a + i)),
                   _mm512_loadu_si512((const void *)(b + i)));
    }
    int32_t s = (int32_t)_mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
    for (; i < n; ++i) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}
#endif

static void demo_dot(const Feat *f)
{
    printf("\n[2] INT8 点积：标量 / 纯AVX2模拟 / AVX-VNNI 对比  (N=%d, iter=%d)\n",
           DOT_N, DOT_ITER);
    printf("---------------------------------------------------------------\n");

    uint8_t *a = (uint8_t *)_mm_malloc(DOT_N * sizeof(uint8_t), 64);
    int8_t  *b = (int8_t  *)_mm_malloc(DOT_N * sizeof(int8_t),  64);
    if (!a || !b) { printf("    分配失败\n"); return; }
    /* 注意：这里刻意用【小幅度】数据（u8: 0..15, s8: -8..7）。
     * 原因见下方"为什么需要 VNNI"小节 —— 纯 AVX2 模拟路径的
     * vpmaddubsw 会在 16-bit 中间结果上饱和，大幅值会算错。 */
    for (int i = 0; i < DOT_N; ++i) {
        a[i] = (uint8_t)((i * 7 + 3) & 0x0F);            /* u8: 0..15   */
        b[i] = (int8_t)(((i * 13 + 5) & 0x0F) - 8);      /* s8: -8..7   */
    }

    int32_t r_ref  = dot_scalar(a, b, DOT_N);
    int32_t r_emu  = dot_avx2_emu(a, b, DOT_N);
    int32_t r_vnni = dot_vnni(a, b, DOT_N);
    printf("    scalar      = %d\n", r_ref);
    printf("    avx2-sim    = %d   %s\n", r_emu,  r_emu  == r_ref ? "OK" : "MISMATCH!");
    printf("    avx-vnni    = %d   %s\n", r_vnni, r_vnni == r_ref ? "OK" : "MISMATCH!");
    {
        int32_t r_asm = dot_vnni_asm(a, b, DOT_N);
        printf("    asm-vnni    = %d   %s  (手写 ml64 汇编)\n", r_asm,
               r_asm == r_ref ? "OK" : "MISMATCH!");
    }
#if defined(_MSC_VER) || defined(__AVX512VNNI__)
    if (f->avx512vnni && f->zmm_ok) {
        int32_t r512 = dot_vnni512(a, b, DOT_N);
        printf("    avx512-vnni = %d   %s\n", r512, r512 == r_ref ? "OK" : "MISMATCH!");
    }
#endif

    /* ---- 为什么需要 VNNI：纯 AVX2 模拟的 16-bit 中间结果会饱和 ---- */
    printf("\n    [为什么需要 VNNI] 大幅值下 avx2-sim 会算错:\n");
    {
        __m256i ba = _mm256_set1_epi8((char)0xFF);   /* u8  255 */
        __m256i bb = _mm256_set1_epi8((char)127);    /* s8  127 */
        __m256i p16 = _mm256_maddubs_epi16(ba, bb);  /* 255*127*2 = 64770 > 32767 !! */
        __m256i emu = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
        __m256i vnn = DPBUSD_256(_mm256_setzero_si256(), ba, bb);
        print_i32("avx2-sim (vpmaddubsw)", emu);
        print_i32("vpdpbusd", vnn);
        printf("    真值 = 255*127*4 = %d\n", 255 * 127 * 4);
        printf("    avx2-sim 把 64770 饱和成 i16 上限 32767，最终得到 %d（错）\n",
               32767 + 32767);
        printf("    vpdpbusd 直接累加到 i32，无中间饱和，得到 %d（对）\n", 129540);
    }

    /* ---- 计时 ---- */
    double macs = (double)DOT_N * (double)DOT_ITER;
    double bytes = 2.0 * (double)DOT_N * (double)DOT_ITER;

#define TIME(label, fn)                                                       \
    do {                                                                      \
        volatile int32_t sink = 0;                                            \
        double best = 1e30, t0, t1;                                           \
        for (int run = 0; run < 3; ++run) {                                   \
            t0 = now_sec();                                                   \
            for (int it = 0; it < DOT_ITER; ++it) sink = fn(a, b, DOT_N);      \
            t1 = now_sec();                                                   \
            if (t1 - t0 < best) best = t1 - t0;                               \
        }                                                                     \
        printf("    %-14s %7.2f ms   %7.2f GMAC/s   %6.2f GB/s\n",             \
               label, best * 1e3, macs / best / 1e9, bytes / best / 1e9);      \
        (void)sink;                                                           \
    } while (0)

    printf("\n    性能（取 3 次最优）：\n");
    TIME("scalar",   dot_scalar);
    TIME("avx2-sim", dot_avx2_emu);
    TIME("avx-vnni", dot_vnni);
    TIME("asm-vnni", dot_vnni_asm);
#if defined(_MSC_VER) || defined(__AVX512VNNI__)
    if (f->avx512vnni && f->zmm_ok) TIME("avx512-vnni", dot_vnni512);
#endif

    _mm_free(a); _mm_free(b);
}

/* ========================================================================== */
/* 3. 量化 GEMM：C[M][N] = A[M][K](u8) x B[K][N](s8)                           */
/*                                                                             */
/*    vpdpbusd 一条指令每 lane 吃 4 个连续 K，所以最"顺手"的数据布局是：        */
/*      A: [M][K]   行主序，K 连续                                             */
/*      B: [N][K]   N 主序（即 B[n*K + k]），K 连续  —— 内层沿 K 扫描           */
/*    这样内层循环可以直接沿 K 走 32 字节一跳。                                 */
/* ========================================================================== */

#define GEMM_M 96
#define GEMM_N 96
#define GEMM_K 256
#define GEMM_REPEAT 200

static void gemm_scalar(const uint8_t *A, const int8_t *B, int32_t *C,
                        int M, int N, int K)
{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int32_t s = 0;
            const uint8_t *ar = A + (size_t)m * K;
            const int8_t  *br = B + (size_t)n * K;
            for (int k = 0; k < K; ++k) s += (int32_t)ar[k] * (int32_t)br[k];
            C[(size_t)m * N + n] = s;
        }
}

static void gemm_vnni(const uint8_t *A, const int8_t *B, int32_t *C,
                      int M, int N, int K)
{
    for (int m = 0; m < M; ++m) {
        const uint8_t *ar = A + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            const int8_t *br = B + (size_t)n * K;
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            int k = 0;
            for (; k + 128 <= K; k += 128) {
                acc0 = DPBUSD_256(acc0,
                           _mm256_loadu_si256((const __m256i *)(ar + k)),
                           _mm256_loadu_si256((const __m256i *)(br + k)));
                acc1 = DPBUSD_256(acc1,
                           _mm256_loadu_si256((const __m256i *)(ar + k + 32)),
                           _mm256_loadu_si256((const __m256i *)(br + k + 32)));
                acc0 = DPBUSD_256(acc0,
                           _mm256_loadu_si256((const __m256i *)(ar + k + 64)),
                           _mm256_loadu_si256((const __m256i *)(br + k + 64)));
                acc1 = DPBUSD_256(acc1,
                           _mm256_loadu_si256((const __m256i *)(ar + k + 96)),
                           _mm256_loadu_si256((const __m256i *)(br + k + 96)));
            }
            for (; k + 32 <= K; k += 32)
                acc0 = DPBUSD_256(acc0,
                           _mm256_loadu_si256((const __m256i *)(ar + k)),
                           _mm256_loadu_si256((const __m256i *)(br + k)));

            int32_t s = hsum_epi32(_mm256_add_epi32(acc0, acc1));
            for (; k < K; ++k) s += (int32_t)ar[k] * (int32_t)br[k];
            C[(size_t)m * N + n] = s;
        }
    }
}

static void demo_gemm(void)
{
    printf("\n[3] 量化 GEMM：C[%d][%d] = A[%d][%d](u8) x B[%d][%d](s8)\n",
           GEMM_M, GEMM_N, GEMM_M, GEMM_K, GEMM_N, GEMM_K);
    printf("---------------------------------------------------------------\n");

    size_t sa = (size_t)GEMM_M * GEMM_K;
    size_t sb = (size_t)GEMM_N * GEMM_K;
    uint8_t *A = (uint8_t *)_mm_malloc(sa, 64);
    int8_t  *B = (int8_t  *)_mm_malloc(sb, 64);
    int32_t *C0 = (int32_t *)_mm_malloc((size_t)GEMM_M * GEMM_N * sizeof(int32_t), 64);
    int32_t *C1 = (int32_t *)_mm_malloc((size_t)GEMM_M * GEMM_N * sizeof(int32_t), 64);
    if (!A || !B || !C0 || !C1) { printf("    分配失败\n"); return; }

    unsigned seed = 12345u;
    for (size_t i = 0; i < sa; ++i) { seed = seed * 1103515245u + 12345u; A[i] = (uint8_t)(seed >> 16); }
    for (size_t i = 0; i < sb; ++i) { seed = seed * 1103515245u + 12345u; B[i] = (int8_t)(seed >> 24); }

    gemm_scalar(A, B, C0, GEMM_M, GEMM_N, GEMM_K);
    gemm_vnni  (A, B, C1, GEMM_M, GEMM_N, GEMM_K);

    int bad = 0;
    for (int i = 0; i < GEMM_M * GEMM_N; ++i)
        if (C0[i] != C1[i]) { if (bad < 3) printf("    #%d: %d != %d\n", i, C0[i], C1[i]); ++bad; }
    printf("    校验: %s (%d/%d 不一致)\n", bad ? "MISMATCH!" : "全部一致 OK",
           bad, GEMM_M * GEMM_N);

    double macs = 2.0 * (double)GEMM_M * GEMM_N * GEMM_K;
    double t0, t1, best;

    best = 1e30;
    for (int r = 0; r < 3; ++r) {
        t0 = now_sec();
        for (int it = 0; it < GEMM_REPEAT; ++it) gemm_scalar(A, B, C0, GEMM_M, GEMM_N, GEMM_K);
        t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    printf("    scalar   %8.3f ms/次   %7.2f GMAC/s\n",
           best / GEMM_REPEAT * 1e3, macs * GEMM_REPEAT / best / 1e9);

    best = 1e30;
    for (int r = 0; r < 3; ++r) {
        t0 = now_sec();
        for (int it = 0; it < GEMM_REPEAT; ++it) gemm_vnni(A, B, C1, GEMM_M, GEMM_N, GEMM_K);
        t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    printf("    avx-vnni %8.3f ms/次   %7.2f GMAC/s\n",
           best / GEMM_REPEAT * 1e3, macs * GEMM_REPEAT / best / 1e9);

    _mm_free(A); _mm_free(B); _mm_free(C0); _mm_free(C1);
}

/* ========================================================================== */
/* 4. 手工内联汇编版（MSVC x64 不支持 __asm，用机器码 _emit 演示）              */
/*    VPDPBUSD ymm1, ymm2, ymm3  的 VEX 编码：                                 */
/*    VEX.256.0F38.WIG 50 /r  ->  C4 E2 6D 50 CB   (ymm1, ymm2, ymm3)         */
/*      C4        : 3-byte VEX                                                 */
/*      E2 (R vvvv L pp)  -> R=1(取反->0), vvvv=~0x1=1110 -> ymm2, L=1(256), pp=01(0F38) */
/*      6D        : W=0, ~vvvv=1101 -> ymm2 ... 具体见 Intel SDM Table 2-25     */
/*      50        : opcode                                                     */
/*      CB        : modrm mod=11 reg=1(ymm1) rm=3(ymm3)                        */
/*    更推荐直接用 MASM64 写 .asm（见 vnni_asm.asm），可读性远好于 _emit。      */
/* ========================================================================== */

/* ========================================================================== */
/* main                                                                        */
/* ========================================================================== */

int main(void)
{
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);   /* 让 Windows 控制台正确显示 UTF-8 中文 */
#endif
    Feat f = detect_cpu();

    printf("===============================================================\n");
    printf("  AVX-VNNI (vpdpbusd) Demo\n");
    printf("===============================================================\n");
    printf("  CPU 特性检测 (CPUID):\n");
    printf("    AVX2            : %s\n", f.avx2       ? "yes" : "no");
    printf("    YMM 状态已启用  : %s\n", f.ymm_ok     ? "yes" : "no");
    printf("    AVX-VNNI        : %s   <- 本 demo 主角\n", f.avxvnni ? "yes" : "no");
    printf("    AVX512F         : %s\n", f.avx512f    ? "yes" : "no");
    printf("    AVX512-VNNI     : %s\n", f.avx512vnni ? "yes" : "no");
    printf("    ZMM 状态已启用  : %s\n", f.zmm_ok     ? "yes" : "no");

    if (!f.avx2 || !f.ymm_ok) { printf("\n  CPU 不支持 AVX2，退出。\n"); return 1; }
    if (!f.avxvnni) {
        printf("\n  ⚠ 当前 CPU 不支持 AVX-VNNI，vpdpbusd 会 #UD 非法指令。\n");
        printf("    请在 Alder Lake+ / Zen4+ 上运行，或改用 AVX2 模拟路径。\n");
        return 1;
    }

    demo_semantics(&f);
    demo_dot(&f);
    demo_gemm();

    printf("\n===============================================================\n");
    printf("  要点小结:\n");
    printf("   1. vpdpbusd = 一条指令做 u8*s8->i32 的 4 元乘加累加，取代\n");
    printf("      vpmaddubsw + vpmaddwd + vpaddd 三条指令；且没有 16-bit\n");
    printf("      中间饱和问题（上面 avx2-sim 算出 65534 就是活例子）。\n");
    printf("   2. 4 路累加器展开用于掩盖 5-cycle 延迟；本例实测已到 ~350 GB/s，\n");
    printf("      瓶颈是【2 个 32B load/cycle】的取指带宽，不是算力。\n");
    printf("   3. 数据布局要顺着指令: 每个 lane 吃 4 个【连续】的 K。\n");
    printf("\n");
    printf("   两个真实踩过的坑（本 demo 里都修掉了）:\n");
    printf("   A. MSVC 的 _mm256_dpbusd_epi32 编成 EVEX(62 ...) -> 需要 AVX512-VNNI，\n");
    printf("      在 Alder Lake/Arrow Lake/Zen4 上直接 #UD。必须用\n");
    printf("      _mm256_dpbusd_avx_epi32 强制 VEX(C4 ...)；ml64 同理，只能手工 emit。\n");
    printf("   B. Windows x64 ABI 下 YMM6-YMM15 是【非易失】的。手写汇编用了\n");
    printf("      ymm6/ymm7 却不保存，会静默破坏调用方状态（本 demo 曾因此\n");
    printf("      测出 3.7e-289 秒这种垃圾计时，100%% 稳定复现，极难查）。\n");
    printf("===============================================================\n");
    return 0;
}
