# Phase 7: E2E Verification - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-21
**Phase:** 7-E2E Verification
**Areas discussed:** 测试函数设计, Swizzle 往返覆盖率, 参考值计算策略, 回归范围 & LLVM 陷阱

---

## 测试函数设计

### SHTEST-01 测试数量

| Option | Description | Selected |
|--------|-------------|----------|
| 扩展 process_shared_2d 增加混合参数 | 添加一个 distributed tensor 参数，证明 shared + distributed 混合 | |
| 用两个 gl.call 串联读写 | kernel 内先 gl.call 写 → barrier → gl.call 读改回 → load() 读回验证 | |
| 两者都做 | 一个 kernel 串联读写，另一个 kernel 混合参数 | ✓ |
| 我来说具体场景 | 用户描述希望的测试场景 | |

**User's choice:** 两者都做

### 形状、dtype、layout 选择

| Option | Description | Selected |
|--------|-------------|----------|
| 1D float32 + 非平凡 swizzle layout | Shape<512>/<256>，非平凡 offset/block bases | |
| 2D float16 + 简单 linear layout | Shape<32,16>，identity SharedLinearLayout | |
| 1D 和 2D 各自独立 | 串联测试用 2D，混合参数测试用 1D | ✓ |
| 你来建议 | 根据 codebase 判断 | |

**User's choice:** 1D 和 2D 各自独立 — 利用现有 process_shared_2d (2D) + 新建 shared_accumulate (1D)

### 混合参数函数功能

| Option | Description | Selected |
|--------|-------------|----------|
| 读 shared + 加 distributed + 写回 shared | shared_accumulate: 读 shared[i] += val | ✓ |
| 读 distributed + 写 shared | 纯写路径 | |
| 读 shared + 返回 distributed | shared→register 方向 | |
| 你来选 | 基于 codebase 可行性 | |

**User's choice:** shared_accumulate — 读 shared[i] + 加 distributed tensor val + 写回 shared

### process_shared_2d 返回值

| Option | Description | Selected |
|--------|-------------|----------|
| 保持 void 返回，通过 load() 读回验证 | 不改函数签名 | ✓ |
| 改为返回 distributed tensor | 测试 shared→register 返回类型推断 | |
| 两者都覆盖 | 串联测试 void+load()，另一个测试返回 tensor | |

**User's choice:** 保持 void 返回，通过 load() 读回验证

---

## Swizzle 往返覆盖率

### 测试深度

| Option | Description | Selected |
|--------|-------------|----------|
| 单个非平凡 swizzle + 独立 basis 验证 | 两个 kernel 各自隔离一个 basis | |
| 单个综合 swizzle + 单 kernel | 一个非平凡 layout，load() 读回对比 | |
| 参数化多 swizzle | parametrize 3-4 种 pattern | ✓ |
| 你来建议 | 基于 D-07 已有证明 | |

**User's choice:** 参数化 — identity, offset-only, block-only, full xor 共 4 种

### 验证方式

| Option | Description | Selected |
|--------|-------------|----------|
| load() 读回 + CPU 参考 | CPU 模拟 swizzle 逻辑计算期望值 | ✓ |
| load() 读回自验证 | 自洽验证 swizzle 正确往返 | |
| 两种都做 | 双重保险 | |

**User's choice:** load() 读回 + CPU 参考

---

## 参考值计算策略

### shared_accumulate 写回验证

| Option | Description | Selected |
|--------|-------------|----------|
| Python 端预计算参考 | pytest 用 torch 预计算 expected 数组 | ✓ |
| Kernel 内 inline 计算 | triton 不支持 kernel 内 assert | |
| 只需要 load() + store 到 output | 不需要 reference | |
| 标准模式就可以 | 跟现有 test pattern 一样 | |

**User's choice:** Python 端 torch 预计算参考值，kernel store 结果到 output，pytest assert_close 对比

---

## 回归范围 & LLVM 陷阱

### 回归测试覆盖面

| Option | Description | Selected |
|--------|-------------|----------|
| 最小范围 | test_extern_call.py (6) + Gluon lit (5) | |
| 扩展 | 10 E2E + 全部 lit | ✓ |
| 完整回归 | 10 E2E + 全部 lit + Phase 4 GPU 测试 | |
| 你来建议 | 根据回归风险判断 | |

**User's choice:** 扩展 — 全部 10 E2E tests (6 原有 + 4 shared_tensor) + 全部 Gluon lit tests (5 原有 + extern-call-shared-args)

### L-01 landmine 验证方式

| Option | Description | Selected |
|--------|-------------|----------|
| 编译 dump PTX 然后 grep | Python 测试中自动 grep ptx | ✓ |
| 写单独 lit test 检查 PTX | GPU 无关 lit test | |
| 仅文档记录，手动抽查 | 不自动化 | |
| 自动化 + 文档 | PTX grep + CONTEXT.md 记录 | |

**User's choice:** 编译 dump PTX 然后 grep — 在共享内存测试后自动验证 `ld.shared`/`st.shared` 出现

---

## the agent's Discretion

- 4 种 parametrized swizzle 的精确 basis 值 — researcher 选择真实的非平凡 base
- `shared_accumulate` 迭代次数 (REG_SIZE vs shape size) — identity layout 使之相等，REG_SIZE 对未来 layout 更通用
- `num_warps=1` vs `num_warps=2` — researcher 决定
- SHTEST-01 test 1 的 scale 参数值 (1.0 或 2.0)

## Deferred Ideas

无 — 讨论在 phase scope 内。已有延期项保持不变（SHRET-01, AUTO-01, FP64-01, PaddedSharedLayout, dynamic shared, auto-barriers）。
