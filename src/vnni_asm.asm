; =============================================================================
;  vnni_asm.asm  ——  手写 AVX-VNNI 汇编：INT8 点积
;
;  汇编:  ml64 /c /Cx /W3 vnni_asm.asm      (产出 vnni_asm.obj)
;
;  x64 调用约定 (Microsoft):
;      RCX = a (const uint8_t*)   RDX = b (const int8_t*)   R8 = n (size_t)
;      返回 EAX (int32_t)
;      易失:       RAX RCX RDX R8 R9 R10 R11        + YMM0-YMM5
;      非易失:     RBX RBP RDI RSI R12-R15          + YMM6-YMM15
;
;  ⚠ 踩过的坑：本函数最初用了 ymm6/ymm7 做源操作数，而 Windows x64 ABI 规定
;    YMM6-YMM15 是【非易失】的，调用方可能把值放在里面。结果没有保存就被覆盖，
;    导致调用方（MSVC /O2 编译的 C 代码）算出一个 ~3.7e-289 的垃圾计时值，
;    而且是 100% 稳定复现 —— 极难查。
;    修复办法：只用 YMM0-YMM5（本函数正是如此），或者用栈保存 YMM6-YMM15。
;    这里用 2 个临时寄存器 ymm4/ymm5 复用，4 个累加器 ymm0-ymm3，刚好不越界。
;
; -----------------------------------------------------------------------------
;  ⚠ 为什么 vpdpbusd 要手工写机器码？
;
;  ml64 / MSVC 对 vpdpbusd 默认输出的是 **EVEX** 编码（AVX512-VNNI）：
;        62 F2 5D 28 50 02
;  这需要 CPU 支持 AVX512-VNNI。在只有 AVX-VNNI、没有 AVX-512 的 CPU
;  （Alder Lake / Arrow Lake / Lunar Lake / Zen4 ...）上会直接 #UD 非法指令。
;  MASM 也不支持 {vex3} 编码限定符（试过前缀/后缀两个位置，均报 A2008/A2009）。
;
;  所以这里直接用 db 手工 emit **VEX** 编码：
;        C4 E2 5D 50 02
;
;  VEX.256.0F38.WIG 50 /r 的 3 字节 VEX 前缀拆解：
;
;     byte1  C4        : 3 字节 VEX 前缀标志
;     byte2  E2        : [7]R̄ [6]X̄ [5]B̄ [4:0]mmmmm
;                        R̄X̄B̄ = 111 (全取反 -> R=X=B=0, 即 reg/rm 都在 ymm0-7)
;                        mmmmm = 00010 -> 操作码映射表 0F38
;     byte3  5D        : [7]W [6:3]v̄v̄v̄v̄ [2]L [1:0]pp
;                        W=0 (WIG)
;                        v̄v̄v̄v̄ = 1011 -> 取反 = 0100 -> vvvv = ymm4 (第 1 源操作数)
;                        L  = 1   -> 256 bit
;                        pp = 01  -> 0F38
;     然后   50        : opcode
;            02        : ModRM  mod=00 reg=000(ymm0) rm=010(rdx)
;
;  本文件用到的 4 条编码（disp8 用于 +20h/+40h/+60h）：
;     vpdpbusd ymm0, ymm4, [rdx]        C4 E2 5D 50 02
;     vpdpbusd ymm1, ymm5, [rdx+20h]    C4 E2 55 50 4A 20
;     vpdpbusd ymm2, ymm4, [rdx+40h]    C4 E2 5D 50 52 40
;     vpdpbusd ymm3, ymm5, [rdx+60h]    C4 E2 55 50 5A 60
;
;  注意第 1/3 条 byte3 相同(5D)，因为 vvvv 都是 ymm4；
;  第 2/4 条都是 55，因为 vvvv 都是 ymm5。只有 ModRM.reg 不同。
; =============================================================================

.CODE

PUBLIC  dot_vnni_asm

; -----------------------------------------------------------------------------
; int32_t dot_vnni_asm(const uint8_t* a, const int8_t* b, size_t n);
;
;   VPDPBUSD ymm1, ymm2, ymm3/m256
;       for j in 0..7:
;           for k in 0..3:
;               ymm1.d[j] += (uint8)ymm2.b[4j+k] * (int8)ymm3.b[4j+k]
;
;   一条指令 = 32 个 u8*s8 乘加 + 8 路 i32 累加，取代
;   vpmaddubsw + vpmaddwd + vpaddd 三条指令。
; -----------------------------------------------------------------------------
dot_vnni_asm PROC

        vpxor       ymm0, ymm0, ymm0           ; acc0 = 0
        vpxor       ymm1, ymm1, ymm1           ; acc1 = 0
        vpxor       ymm2, ymm2, ymm2           ; acc2 = 0
        vpxor       ymm3, ymm3, ymm3           ; acc3 = 0
        xor         r10d, r10d                 ; 标量尾部累加器

        ; ---------- 主循环：每次处理 4 x 32 = 128 字节 ----------
        mov         r9, r8
        shr         r9, 7                      ; r9 = n / 128
        test        r9, r9
        jz          short @@tail32

        ALIGN       16
@@loop128:
        vmovdqu     ymm4, ymmword ptr [rcx +  0]
        db          0C4h, 0E2h, 05Dh, 050h, 002h          ; vpdpbusd ymm0, ymm4, [rdx]

        vmovdqu     ymm5, ymmword ptr [rcx + 32]
        db          0C4h, 0E2h, 055h, 050h, 04Ah, 020h    ; vpdpbusd ymm1, ymm5, [rdx+20h]

        vmovdqu     ymm4, ymmword ptr [rcx + 64]          ; 复用 ymm4
        db          0C4h, 0E2h, 05Dh, 050h, 052h, 040h    ; vpdpbusd ymm2, ymm4, [rdx+40h]

        vmovdqu     ymm5, ymmword ptr [rcx + 96]          ; 复用 ymm5
        db          0C4h, 0E2h, 055h, 050h, 05Ah, 060h    ; vpdpbusd ymm3, ymm5, [rdx+60h]

        add         rcx, 128
        add         rdx, 128
        dec         r9
        jnz         short @@loop128

        ; ---------- 尾部：32 字节块 ----------
@@tail32:
        mov         r9, r8
        and         r9, 127                    ; 剩余字节数 (n % 128)
        shr         r9, 5                      ; 32 字节块数
        test        r9, r9
        jz          short @@reduce

@@loop32:
        vmovdqu     ymm4, ymmword ptr [rcx]
        db          0C4h, 0E2h, 05Dh, 050h, 002h          ; vpdpbusd ymm0, ymm4, [rdx]
        add         rcx, 32
        add         rdx, 32
        dec         r9
        jnz         short @@loop32

        ; ---------- 归约：4 个累加器 -> 1 个 int32 ----------
@@reduce:
        vpaddd      ymm0, ymm0, ymm1
        vpaddd      ymm2, ymm2, ymm3
        vpaddd      ymm0, ymm0, ymm2

        vextracti128 xmm1, ymm0, 1             ; 高 128 位
        vpaddd      xmm0, xmm0, xmm1           ; 低 + 高 -> 4 个 int32
        vpshufd     xmm1, xmm0, 04Eh           ; 交换两个 64-bit 半区
        vpaddd      xmm0, xmm0, xmm1           ; -> 2 个 int32
        vpshufd     xmm1, xmm0, 0B1h           ; 交换两个 32-bit 半区
        vpaddd      xmm0, xmm0, xmm1           ; -> 1 个 int32
        vmovd       eax, xmm0                  ; EAX = 向量部分结果

        ; ---------- 标量尾部：< 32 字节 ----------
        mov         r9, r8
        and         r9, 31                     ; n % 32
        test        r9, r9
        jz          short @@done

@@loop1:
        movzx       r11d, byte ptr [rcx]       ; a[i] 零扩展 (u8)
        movsx       r8d,  byte ptr [rdx]       ; b[i] 符号扩展 (s8)
        imul        r11d, r8d
        add         r10d, r11d
        inc         rcx
        inc         rdx
        dec         r9
        jnz         short @@loop1

@@done:
        add         eax, r10d                  ; 加上标量尾部
        vzeroupper                             ; 清掉 YMM 高位，避免 AVX/SSE 过渡惩罚
        ret

dot_vnni_asm ENDP

END
