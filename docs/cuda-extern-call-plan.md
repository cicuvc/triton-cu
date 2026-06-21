# CUDA C++ Extern Call — 实现方案

## 目标

实现 `gl.call(src_path, func, *args, result_layout=None)`，允许在 Triton/Gluon kernel 中调用外部 CUDA C++ `__device__` 函数，支持超越 elementwise 的操作（如 warp shuffle）。

## 总体架构

```
Python DSL: gl.call("tt_plugin.cu", "elementwise_add", tensor_a, tensor_b, result_layout=...)
  │
  ▼ GluonSemantic / GluonOpBuilder → 创建 TTIR/TTGIR op
  │
TTGIR: %r = ttg.extern_call(%a, %b) {
         symbol = "elementwise_add",
         libpath = "/path/to/tt_plugin.cu"
       } : (tensor<...xT, #enc>, tensor<...xT, #enc>) -> tensor<...xT, #enc>
  │    ↑ SameOperandsAndResultEncoding → layout 传播与 elementwise 一致
  │    
  ▼ TTGIR布局推断 (现有pass,无需改动)
  │  ResolveAutoEncodings / InferCoalescedEncodings → 具体layout已确定
  │
  ▼ ConvertTritonGPUToLLVM (ExternCallOpConversion)
  │  1. 从input的RankedTensorType::getEncoding()提取LinearLayout信息
  │  2. 序列化layout metadata到module attribute (ttg.extern_call_layouts)
  │  3. 将tensor operands提升为LLVM struct (与CallOpConversion相同)
  │  4. 生成LLVM::CallOp @__triton_ext_<symbol>_<hash>
  │
  ▼ mlir::translateModuleToLLVMIR → llvm::Module
  │
  ▼ make_llir() in compiler.py
  │  1. 读取module attr中的layout metadata
  │  2. 用libclang parse .cu文件 → 理解template签名
  │  3. 根据layout生成具体的C++类型声明
  │  4. 生成wrapper函数（带具体类型，让clang完成template deduction）
  │  5. libclang compile → LLVM bitcode
  │  6. 将.bc加入options.extern_libs → llvm.link_extern_libs()
  │
  ▼ 继续: LLVM optimize → ptx → cubin
```

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| IR op 所在 dialect | `ttg` (TritonGPU) | layout encoding 在 ttg tensor type 上；op 存在于 layout 已解析的阶段 |
| Result layout | 默认与第一个input相同（SameOperandsAndResultEncoding） | `elementwise_add` / `intra_warp_add_sibling` 等返回同layout的tensor |
| 用户指定result layout | 可选参数 `result_layout=`，emit `convert_layout` after `extern_call` | 灵活处理布局不同的情况 |
| Layout传播 | 现有 `inferLayout()` 引擎自动处理 | `SameOperandsAndResultEncoding` trait 使op成为pass-through |
| NVGPU only | 32 threads/warp, lane编码5 bit | 匹配 `tt_plugin.cu` 的假设；AMD有64-thread wavefront暂不支持 |
| libclang | 自编译LLVM (commit `62b7cf96`) | 避免版本不兼容；in-process解析+编译 |
| Wrapper命名 | `__triton_ext_<symbol>_<layout_hash>` | 不同layout的同一function可共存 |
| Cache | 实现后缓存 .bc per `(libpath, symbol, layout_hash)` | 复用 `build.py` 的cache manager |

## 实现步骤

### Step 0: 构建 libclang ✓ (已完成)

- LLVM source: `/home/cicuvc/cs/project/llvm-project` (branch `triton-base`, commit `62b7cf96`)
- 构建配置: `cmake -G Ninja -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" -DCMAKE_BUILD_TYPE=RelWithDebInfo`
- 产物: `lib/libclang.so.23.0.0git`, `bin/clang`
- 安装路径: `/home/cicuvc/cs/project/llvm-project/install`
- Python bindings: `pip install clang` + `Config.set_library_file(...)`

### Step 1: 定义 TTGIR Op

**文件: `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`**

```tablegen
def TTG_ExternCallOp : TTG_Op<"extern_call", [
    SameOperandsAndResultEncoding,        // layout pass-through
    SameVariadicOperandSize,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Call an external CUDA C++ __device__ function";
  let arguments = (ins
    Variadic<TT_Tensor>:$inputs,
    StrAttr:$symbol,      // function name, e.g. "elementwise_add"
    StrAttr:$libpath      // path to .cu source file
  );
  let results = (outs Variadic<TT_Tensor>:$results);
  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $results)
  }];
  let hasVerifier = 1;
}
```

注意：需要同时更新 `TritonGPUOpFolders`、`TritonGPUDialect.cpp` 中的 parse/print、以及 `TritonGPUCanonicalize` 的 dispatch。

### Step 2: Lowering Pattern (TTGIR → LLVM)

**新建文件: `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp`**

```cpp
struct ExternCallOpConversion 
    : public ConvertOpToLLVMPattern<ExternCallOp> {
  
  LogicalResult matchAndRewrite(ExternCallOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *module = op->getParentOfType<ModuleOp>();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    
    // === 1. 提取layout信息（在promoteOperands之前） ===
    SmallVector<LayoutMetadata> inputLayouts;
    for (auto operand : op.getInputs()) {
      auto tensorTy = cast<RankedTensorType>(operand.getType());
      auto encoding = tensorTy.getEncoding();
      auto shape = tensorTy.getShape();
      LinearLayout ll = triton::gpu::toLinearLayout(shape, encoding);
      
      // 提取: shape, num_warps, reg/lane/warp bases
      LayoutMetadata meta;
      meta.shape = shape;  // 如 [32]
      meta.elementType = tensorTy.getElementType();  // 如 f32
      meta.numWarps = ll.getInDimSize(StringAttr::get(ctx, "warp"));
      
      // 提取basis vectors
      meta.regBases = extractBases(ll, kRegister);
      meta.laneBases = extractBases(ll, kLane);
      meta.warpBases = extractBases(ll, kWarp);
      inputLayouts.push_back(meta);
    }
    
    // === 2. 序列化layout metadata到module attribute ===
    auto hash = computeHash(op.getSymbol(), inputLayouts);
    auto mangledName = "__triton_ext_" + op.getSymbol().str() + "_" + hash;
    storeLayoutMetadata(module, mangledName, op.getLibpath(), 
                        op.getSymbol(), inputLayouts);
    
    // === 3. 将tensor operands提升为LLVM struct ===
    auto promotedOperands = getTypeConverter()->promoteOperands(
        loc, op->getOperands(), adaptor.getOperands(), rewriter);
    // 附加shared/scratch pointers (如CallOpConversion)
    appendScratchPointers(op, promotedOperands, rewriter);
    
    // === 4. 声明或获取外部LLVM function ===
    auto resultTypes = getLLVMResultTypes(op);
    auto funcType = LLVM::LLVMFunctionType::get(
        getReturnType(resultTypes), getOperandTypes(promotedOperands));
    auto funcOp = appendOrGetExternFuncOp(
        rewriter, op, mangledName, funcType, 
        op.getLibnameAttr(), op.getLibpathAttr());
    
    // === 5. 生成LLVM call ===
    auto callOp = LLVM::CallOp::create(rewriter, loc, funcOp, promotedOperands);
    
    // === 6. 解包result struct ===
    auto results = unpackResults(op, callOp, rewriter);
    rewriter.replaceOp(op, results);
    return success();
  }
};
```

**关键辅助函数:**
- `extractBases(LinearLayout, inDim)` — 遍历inDim的每个basis position，提取对应的out-dim tuple值
- `computeHash(symbol, layouts)` — SHA256 of layout specs，用于wrapper命名
- `storeLayoutMetadata(module, ...)` — 将layout spec存入 `module->setAttr("ttg.extern_call_specs", dictAttr)`

**注册 pattern:**

文件: `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp`
```cpp
// 在 patterns 集合中添加（约line 162附近）:
populateExternCallOpToLLVMPatterns(typeConverter, patterns, targetInfo, benefit);
```

文件: `include/triton/Conversion/TritonGPUToLLVM/Passes.h`
```cpp
void populateExternCallOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, int benefit);
```

### Step 3: Python IR Builder (Gluon)

**文件: `python/src/gluon_ir.cc`**

在 `GluonOpBuilder` 中添加:
```cpp
.def("create_extern_call",
    [](GluonOpBuilder &builder, const std::string &libpath,
       const std::string &symbol, std::vector<ir::value> args,
       std::vector<ir::type> resultTypes) -> std::vector<ir::value> {
      // 创建 ttg.extern_call op
      // 使用 TritonOpBuilder 中已有的 createOp 基础设施
      auto op = builder.create<ExternCallOp>(
          builder.getStrAttr(symbol),
          builder.getStrAttr(libpath),
          /*inputs=*/unwrap(args),
          /*resultTypes=*/unwrap(resultTypes));
      return wrap(op->getResults());
    })
```

同时需要在 `python/src/ir.cc` 的 `TritonOpBuilder` 中注册 op 创建基础设施（或确保 `ExternCallOp` 通过通用 create 路径可用）。

### Step 4: Gluon Python DSL

**文件: `python/triton/experimental/gluon/language/_semantic.py`**

```python
class GluonSemantic:
    def call_extern(self, src_path, func, args, result_layouts=None):
        """调用外部 CUDA C++ __device__ 函数"""
        arg_handles = [a.handle for a in args]
        
        if result_layouts is None:
            # 默认: result type 使用第一个input的type（SameOperandsAndResultEncoding）
            result_types = [args[0].type]
        else:
            result_types = [
                distributed_type(args[0].dtype, args[0].shape, layout)
                for layout in result_layouts
            ]
        
        result_ir_types = [rt._to_ir(self.builder) for rt in result_types]
        result_handles = self.builder.create_extern_call(
            str(src_path), func, arg_handles, result_ir_types)
        
        results = [tensor(h, rt) for h, rt in zip(result_handles, result_types)]
        
        if result_layouts is not None:
            # 用户指定了输出layout → 插入 convert_layout
            results = [self.convert_layout(r, layout) 
                       for r, layout in zip(results, result_layouts)]
        
        return results[0] if len(results) == 1 else tuple(results)
```

**文件: `python/triton/experimental/gluon/language/_core.py`**

```python
@builtin
def call(src_path, func, *args, result_layout=None, _semantic=None):
    """
    调用 .cu 文件中的 __device__ 函数。
    
    参数:
      src_path: .cu 源文件路径 (PathLike)
      func: __device__ 函数名 (str)
      *args: Tensor 参数
      result_layout: 可选的输出layout。不指定则与第一个input相同。
    
    示例:
      result = gl.call("tt_plugin.cu", "elementwise_add", a, b)
      result = gl.call("tt_plugin.cu", "intra_warp_add_sibling", x,
                        result_layout=my_layout)
    """
    from pathlib import Path
    src_path = Path(src_path)
    if not src_path.is_absolute():
        # 相对于当前工作目录或相对于caller file
        raise ValueError("src_path must be absolute")
    
    tensors = [_semantic.to_tensor(a) for a in args]
    return _semantic.call_extern(src_path, func, tensors, 
                                  result_layouts=[result_layout] if result_layout else None)
```

### Step 5: CUDA JIT Compilation (make_llir)

**文件: `third_party/nvidia/backend/compiler.py`**

在 `CUDABackend.make_llir()` 中添加（MLIR passes 之后、llvm.to_module 之前）:

```python
def make_llir(self, src, metadata, options, capability):
    mod = src
    pm = ir.pass_manager(mod.context)
    # ... 现有的passes（包括 add_to_llvmir）...
    pm.run(mod, 'make_llir')
    
    # ==== NEW: 处理 extern_call ====
    # 读取 ExternCallOpConversion 写入的 module attribute
    extern_specs = self._collect_extern_call_specs(mod)
    
    # 将每个 spec 编译为 .bc 并加入 extern_libs
    for spec in extern_specs:
        bc_path = self._jit_compile_cu(spec, capability)
        if "extern_libs" not in metadata:
            metadata["extern_libs"] = []
        metadata["extern_libs"].append((spec.mangled_name, bc_path))
    
    # 现有的 LLVM 翻译和链接流程
    llvm_mod = llvm.to_module(mod, context)
    # ... attach_datalayout, features ...
    
    if metadata.get("extern_libs"):
        paths = [path for _, path in metadata["extern_libs"]]
        llvm.link_extern_libs(llvm_mod, paths)
    
    # ... optimize, extract metadata ...
```

**`_collect_extern_call_specs()`:**
```python
def _collect_extern_call_specs(self, mod):
    """从MLIR module attribute读取extern_call layout metadata"""
    specs = {}
    if mod.has_attr("ttg.extern_call_specs"):
        import json
        specs = json.loads(str(mod.get_attr("ttg.extern_call_specs")))
    return specs
```

**`_jit_compile_cu()` — 核心: libclang 编译流程:**

```python
def _jit_compile_cu(self, spec, capability):
    """
    用libclang将 .cu 编译为 LLVM bitcode。
    
    流程:
      1. Parse: 用 libclang 解析 .cu → 理解 template 签名
      2. Generate: 根据 layout metadata 生成具体的 C++ 类型
      3. Compile: 将 generated source + 原始 .cu 编译为 .bc
    """
    import clang.cindex
    from clang.cindex import CursorKind, TypeKind
    
    libpath = spec["libpath"]
    symbol = spec["symbol"]
    layouts = spec["inputs"]  # list of {shape, dtype, num_warps, reg_bases, lane_bases, warp_bases}
    
    # === 1. Parse ===
    tu = self._parse_cu_file(libpath)
    func_node = self._find_template_function(tu, symbol)
    
    # === 2. Generate concrete C++ types ===
    generated_source = self._generate_cu_wrapper(libpath, symbol, layouts)
    
    # === 3. Compile to .bc ===
    # 使用 clang binary 编译生成的 source + 原始 .cu
    # 或使用 libclang API 直接生成 LLVM bitcode
    
    bc_path = self._compile_to_bc(generated_source, capability)
    return bc_path
```

**`_generate_cu_wrapper()` — 生成 wrapper 源码:**

```python
def _generate_cu_wrapper(self, libpath, symbol, layouts):
    """生成 CUDA wrapper 源码，实例化对应的 template"""
    abs_libpath = os.path.abspath(libpath)
    
    lines = []
    lines.append(f'#include "{abs_libpath}"')
    lines.append('')
    
    # 为每个不同的input layout生成对应的类型别名
    for i, layout in enumerate(layouts):
        shape = layout["shape"]       # 如 [32]
        num_warps = layout["num_warps"]  # 如 4
        dtype = layout["dtype"]       # 如 "f32"
        
        cpp_dtype = self._triton_dtype_to_cpp(dtype)  # f32 → float
        
        # Shape<32>
        lines.append(f'using S{i} = Shape<{",".join(map(str,shape))}>;')
        
        # TensorLayout<Shape<32>, 4>
        lines.append(f'using TL{i} = TensorLayout<S{i}, {num_warps}>;')
        
        # Layout<REGS, LANES, WARPS>
        reg_bases = self._format_basis_group(layout["reg_bases"])
        lane_bases = self._format_basis_group(layout["lane_bases"])
        warp_bases = self._format_basis_group(layout["warp_bases"])
        lines.append(f'using L{i} = TL{i}::Layout<')
        lines.append(f'    TL{i}::BasisGroup<{len(reg_bases)}>{{{reg_bases}}},')
        lines.append(f'    TL{i}::BasisGroup<5>{{{lane_bases}}},')
        lines.append(f'    TL{i}::BasisGroup<{len(warp_bases)}>{{{warp_bases}}}')
        lines.append(f'  >;')
        
        # Tensor<float, Shape<32>, Layout<...>>
        lines.append(f'using TT{i} = Tensor<{cpp_dtype}, S{i}, L{i}>;')
        lines.append('')
    
    # 生成 wrapper 函数
    arg_list = ', '.join(f'const TT{j}& arg{j}' for j in range(len(layouts)))
    lines.append(f'__device__ TT0 __triton_ext_{symbol}_{self._hash(layouts)}({arg_list}) {{')
    call_args = ', '.join(f'arg{j}' for j in range(len(layouts)))
    lines.append(f'    return {symbol}({call_args});')
    lines.append('}')
    
    return '\n'.join(lines)
```

**`_compile_to_bc()`:**

```python
def _compile_to_bc(self, source, capability):
    """用 clang binary 编译 CUDA source → LLVM bitcode"""
    import tempfile, subprocess, os
    
    sm = f'sm_{capability // 10}{capability % 10}'
    
    with tempfile.NamedTemporaryFile(suffix='.cu', mode='w', delete=False) as f:
        f.write(source)
        cu_path = f.name
    
    bc_path = cu_path + '.bc'
    clang = '/home/cicuvc/cs/project/llvm-project/install/bin/clang'
    
    # CUDA headers and clang resource dir
    cuda_inc = os.path.join(os.path.dirname(__file__), 'include')
    resource_dir = '/home/cicuvc/cs/project/llvm-project/install/lib/clang/23'
    
    cmd = [
        clang,
        '-x', 'cuda',
        '-std=c++20',
        '--cuda-device-only',
        '--cuda-gpu-arch=' + sm,
        '-nocudalib',
        '-D__device__=__attribute__((device))',
        '-D__global__=__attribute__((global))',
        '-I', cuda_inc,
        '-resource-dir', resource_dir,
        '-emit-llvm', '-c', cu_path, '-o', bc_path,
    ]
    subprocess.run(cmd, check=True)
    
    return bc_path
```

### Step 6: Layout Metadata 格式

Module attribute `ttg.extern_call_specs` 的 JSON 格式:

```json
{
  "__triton_ext_elementwise_add_a1b2c3d4": {
    "libpath": "/path/to/tt_plugin.cu",
    "symbol": "elementwise_add",
    "inputs": [
      {
        "shape": [32],
        "dtype": "f32",
        "num_warps": 4,
        "reg_bases": [[1], [2]],
        "lane_bases": [[4], [8], [16], [32], [64]],
        "warp_bases": [[128], [256]]
      }
    ]
  }
}
```

## 文件改动清单

| # | 文件 | 改动 |
|---|------|------|
| 0 | `scripts/build-llvm-project.sh` | (可选) 添加 libclang build flag说明 |
| 1 | `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` | 新增 `ttg.extern_call` op 定义 |
| 2 | `include/triton/Dialect/TritonGPU/IR/TritonGPUDialect.cpp` | parse/print 支持 |
| 3 | `lib/Conversion/TritonGPUToLLVM/ExternCallOpToLLVM.cpp` | 新建 lowering pattern |
| 4 | `lib/Conversion/TritonGPUToLLVM/CMakeLists.txt` | 添加新 .cpp |
| 5 | `include/triton/Conversion/TritonGPUToLLVM/Passes.h` | 声明 `populateExternCallOpToLLVMPatterns` |
| 6 | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp` | 注册 pattern |
| 7 | `python/src/gluon_ir.cc` | `create_extern_call` builder 方法 |
| 8 | `python/triton/experimental/gluon/language/_semantic.py` | `call_extern()` |
| 9 | `python/triton/experimental/gluon/language/_core.py` | `gl.call()` builtin |
| 10 | `third_party/nvidia/backend/compiler.py` | `make_llir()` 集成: layout提取 + JIT编译 + 链接 |
| 11 | `python/triton/tools/cuda_jit.py` (新建) | libclang wrapper: .cu → .bc |
| 12 | 测试 | Python test (Gluon kernel 调用 tt_plugin.cu 中的函数) |

## 未解决问题

1. **clang binary vs libclang 编译**: 当前方案用 clang binary (subprocess) 编译；未来可改为 libclang in-process 编译（需要理解 libclang 的 codegen API）。

2. **多输出函数**: 暂不支持返回多个 tensor 的 CUDA 函数；需要扩展 `result_layouts` 参数和类型推导。

3. **非模板函数**: 当前 `tt_plugin.cu` 中的函数都是模板。如果用户提供了非模板的 `__device__` 函数，不需要类型推导，直接 link 即可。

4. **Triton (非Gluon) 路径**: 需要定义 `tt.extern_call` → `ttg.extern_call` 的 conversion pattern in `TritonToTritonGPUPass.cpp`。

5. **AMD 支持**: 当前只支持 NVGPU (32 threads/warp, 5-bit lane)。AMD 需要不同 `tt_plugin.cu` 的变体。
