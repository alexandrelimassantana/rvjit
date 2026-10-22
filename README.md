# The RISC-V Just-In-Time Library

The RISC-V JIT library (`rvjit`) is a header-only C++11 library that extends
RISC-V dynamic code generators, such as
[Xbyak_riscv](https://github.com/herumi/xbyak_riscv), with reusable components
that emit the recurring patterns of micro-kernels: register blocks, unrolled
loops, and memory and arithmetic streams. A micro-kernel is written as a
composition of these components. When it is generated, each component chooses
how to emit its code from the target system, the problem and its own policy,
so a single source serves many systems.

The project has two goals: improve the reusability of low-level RISC-V code,
and define a low-level interface for vendor-agnostic micro-kernels.

## Motivation

Development of `rvjit` started at the Barcelona Supercomputing Center (BSC) in
2021 to instantiate linear algebra runtimes for RISC-V long vector
accelerators. It has since become a tool for vendor-agnostic code generation,
motivated by empirical findings [[1]](#ref-ppopp23) [[2]](#ref-ipdps26) that
the portability of JIT-generated kernels requires adapting the emission logic
of specific components.

`rvjit` therefore places each performance-sensitive decision inside a
component with a well-defined interface, which acts as an intermediate
representation of the concept it describes. ISA extensions are concrete
implementations of one or more components. Components that supply tuning
variables, such as unrolling factors, register blocking and LMUL grouping,
absorb micro-architectural differences within one ISA. As a last resort, an
algorithm can be recomposed for a specific target.

1. <a id="ref-ppopp23"></a>A. de Limas Santana, A. Armejach and M. Casas.
   *Efficient Direct Convolution Using Long SIMD Instructions*. Proceedings of
   the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel
   Programming (PPoPP '23), pp. 342–353, 2023.
   [doi:10.1145/3572848.3577435](https://doi.org/10.1145/3572848.3577435)
2. <a id="ref-ipdps26"></a>A. de Limas Santana, A. Armejach and M. Casas.
   *Just-in-Time Convolution and GEMM Code Generation for SIMD Architectures*.
   2026 IEEE International Parallel and Distributed Processing Symposium
   (IPDPS), pp. 1105–1119, 2026.
   [doi:10.1109/IPDPS65963.2026.00093](https://doi.org/10.1109/IPDPS65963.2026.00093)

## Install

`rvjit` is header-only. The repository is cloned together with its
Xbyak_riscv submodule:

```bash
git clone --recursive https://github.com/alexandrelimassantana/rvjit.git
```

and added to a CMake project with:

```cmake
add_subdirectory(rvjit)
target_link_libraries(my_kernels PRIVATE rvjit::rvjit)
```

An installed copy is found with `find_package(rvjit)` instead. Kernels include
`rvjit/rvjit.hpp`.

## Usage

A kernel is an ordinary `Xbyak_riscv::CodeGenerator`, and hand-written
Xbyak_riscv instructions mix freely with the components.
[`rvv::base_programming_model_t`](rvjit/implementation/rvv/base_programming_model.hpp)
creates every component with its default policies and exposes each one by
name (`pm.arith`, `pm.cf`, `pm.pool`, ...). A kernel that needs other policies
creates the components itself, in the same order. The snippets below go inside
this constructor:

```cpp
#include "rvjit/rvjit.hpp"

using namespace rvjit;

struct kernel_t : public Xbyak_riscv::CodeGenerator {
    explicit kernel_t(const rvv::vpu_t &model) {
        using namespace Xbyak_riscv;

        // The components never draw a0-a2, which hold the arguments
        auto regs = rvv::buffer::architecture_t::default_list();
        regs.integer.without({a0, a1, a2});
        rvv::base_programming_model_t pm(*this, model, regs);

        // ... snippets ...

        ret();
        ready();
    }
};

// A processor with 128-bit vectors and a 32 KiB cache (64-byte lines,
// 4-cycle latency); another model generates another kernel from this source
rvv::vpu_t model;
model.from_params(128, 32768, 64, 4);
kernel_t kernel(model);
```

### Constant arithmetic

Scalar arithmetic with a constant operand, folded into the cheapest
instruction sequence. An immediate that does not fit in 12 bits needs a
register to hold it. Interface:
[`const_arithmetic.hpp`](rvjit/interface/isa/operation/const_arithmetic.hpp).

```cpp
pm.arith.add(a1, a1, 16);                // addi
pm.arith.add(a1, a1, const_t(4096, t0)); // li t0, 4096; add
pm.arith.mul(a2, a2, 8);                 // slli
```

### Control flow

`if_`, `if_else`, `while_` and `switch_case` over branch conditions built by
`branch_t` (`eq`, `nez`, `gtz`, ...). Interface:
[`control_flow.hpp`](rvjit/interface/isa/operation/control_flow.hpp).

```cpp
using control_flow::branch_t;
pm.cf.if_(branch_t::nez(a2), [&] { addi(a1, a1, 1); });
pm.cf.while_(branch_t::gtz(a2), [&] { addi(a2, a2, -1); });
```

### Buffer pool and stack

The pool hands out integer, float and vector registers. What a scope draws is
free again when the scope ends. The stack saves the callee-saved registers the
pool handed out. Interfaces:
[`buffer_pool.hpp`](rvjit/interface/isa/buffer/buffer_pool.hpp),
[`stack.hpp`](rvjit/interface/isa/buffer/stack.hpp).

```cpp
pm.pool.scope([&] {
    const Reg tmp(pm.pool.integers().pop());
    li(tmp, 42);
    return shortfall_t {};
});
pm.stack.preserve(); // at the kernel's entry
pm.stack.restore();  // before ret()
```

### Memory

Loads and stores described by type and form (scalar or vector; unit-stride,
strided or indexed). The backend picks the instruction. Interface:
[`memory.hpp`](rvjit/interface/isa/operation/memory.hpp).

```cpp
memory::op_t vload = memory::op_t().vector().load().as_type(type_t::f32);
vload.data.name = reg_name(v8);
vload.addr = pointer_t(a0, 0);
pm.isa().memory().load(vload); // vle32.v v8, (a0)
```

### Arithmetic

Multiply-accumulate, add, mul, zero, widen, narrow, extend and convert,
described by operand types and forms. `supports_*` tells whether the target
can emit an operation. Interface:
[`arithmetic.hpp`](rvjit/interface/isa/operation/arithmetic.hpp).

```cpp
const auto fma = arithmetic::op_t::uniform(type_t::f32, form_t::vector)
                         .names(reg_name(v16), reg_name(v8), reg_name(v12));
pm.isa().arithmetic().macc(fma); // vfmacc.vv v16, v8, v12
```

### Loops

Data-parallel and reduction loops over literal, runtime or bounded trip
counts. The loop picks its schedule (straight-line, unrolled with a tail, or a
switch) and calls the body once per block with the unrolling factor of that
block. Interfaces: [`loopgen.hpp`](rvjit/interface/loop/loopgen.hpp),
[`plan.hpp`](rvjit/interface/loop/plan.hpp).

```cpp
loop::loop_t *l = pm.loops.new_loop(loop::plan_t::data_parallel(
        size_parameter_t::less_eq(a2, 64), /*max_ur=*/4, /*granularity=*/1));
l->allocate();
l->generate([&](int ur) { addi(a1, a1, ur); });
```

### Streams

A memory operand stream walks a multi-dimensional space, loading into or
storing from a ring of registers. It picks a pointer strategy that fits the
registers left. The combinators nest and interleave streams across unrolled
loops. Interfaces:
[`memory_operand.hpp`](rvjit/interface/stream/memory/memory_operand.hpp),
[`iterator/`](rvjit/interface/stream/iterator/),
[`combinator.hpp`](rvjit/interface/stream/combinator.hpp).

```cpp
pm.pool.set_vector_group_size(1);
stream::iterator::space_t space;
space.axis(4, const_t(64)); // 4 rows, 64 bytes apart

stream::memory_operand::spec_t spec;
spec.op = vload;
spec.origin = pointer_t(a0, 0);
spec.space = space;
spec.addressing = stream::iterator::addressing_t::base_only;

stream::memory_operand::variant_t variant;
variant.ring = buffer_map_t({1, 1}); // one register, drawn from the pool

auto *rows = pm.memops.create(spec, variant);
if (rows->is_valid())
    for (int r = 0; r < 4; ++r)
        rows->iteration({r}); // load row r
```

### Matrix multiplication

`linalg::matmul::matmul_t` composes all of the above into a matrix
multiplication micro-kernel with conversions, bias and post-ops. The
processor model decides its blocking, unrolling and register use. The kernel
of the matrix multiplication benchmark,
[`test/bench/matmul/matmul_kernel.hpp`](test/bench/matmul/matmul_kernel.hpp),
exercises every option. Interface:
[`matmul.hpp`](rvjit/interface/linalg/matmul.hpp).

## Build

The library itself needs no build; the build produces the test suite and the
benchmarks. It requires CMake 3.14 or later and, on hosts other than RISC-V, a
RISC-V cross-compiler and QEMU:

```bash
sudo apt install g++-14-riscv64-linux-gnu qemu-user
git submodule update --init
.github/automation/build.sh build
```

Main CMake options:

| Option | Default | Effect |
|---|---|---|
| `RVJIT_BUILD_TESTS` | `ON` | Build the test suite |
| `RVJIT_BUILD_BENCH` | `ON` | Build the benchmarks |
| `RVJIT_DEBUG` | `ON` | Explain why a kernel could not be generated (printed when `RVJIT_VERBOSE=2`) |
| `RVJIT_BUILD_DOCS` | `OFF` | Add the `rvjit_docs` target for the Doxygen API reference |

## Test

```bash
.github/automation/test.sh build
```

The tests generate small kernels, run them, and check the results they
compute. On a machine that is not RISC-V, they run under `qemu-riscv64`. The
`f16` and `bf16` paths need QEMU 10.2 or later.

`test/bench/` contains benchmarks for the matrix multiplication engine and the
memory streams. Each benchmark has a correctness mode and a timing mode, and
the scripts in `test/bench/scripts/` run them over many configurations.
The performance benchmarks are experimental and not statistically rigorous.

## Documentation

The API reference is generated from the headers with Doxygen
(`RVJIT_BUILD_DOCS=ON`, target `rvjit_docs`, output in `build/docs/html`) and
published to the repository's GitHub Pages on every push to `main`.

## Charter

The repository hosts the `rvjit` component system: the interface of each
component and its concrete implementations for ratified RISC-V ISAs, such as
RV64 and the vector extension (RVV).

The interface is the central design concern of the project. The community is
encouraged to discuss the API design, so that custom ISA extensions can be
supported with minimal duplication of micro-kernel logic.

`rvjit` follows Semantic Versioning over its public interface, the headers in
`rvjit/interface/` and the component implementations that host projects
construct. Changes that break that interface, drop an ISA implementation or
raise the language or toolchain requirements increment the major version.
New components, implementations for newly ratified extensions, new kernel
compositions and changes to the generated code that preserve the interface
increment the minor version. Corrections that leave the interface and the
generated code unchanged increment the patch version. Interface elements are
deprecated for at least one minor release before their removal. Before
version 1.0, minor releases may also break the interface. Releases are
tagged on `main`, and host projects pin a tagged release.

`rvjit` is intended to be used as a component of host projects, such as
oneDNN, which adopt it as an optional, RV64-only dependency and keep their own
data types and kernel dispatch.

## Contributing

- New components and API changes are discussed in an issue before their
  implementation.
- Changes cross-compile to RV64 and are formatted with `clang-format`.
- New behavior comes with a test in `test/gtest/` that generates a small
  kernel, runs it and checks the values it computes, including edge-case
  parameters.
- Changes to code-emitting components are validated under `qemu-riscv64`
  across the VLENs, extensions (`zvfh`, `zvfbfwma`) and shapes they affect.
- Commit messages follow `<scope>: <short description>`, in the imperative
  mood.
- Contributions are licensed under the Apache License 2.0.

## License

`rvjit` is released under the [Apache License 2.0](LICENSE).

It uses [Xbyak_riscv](https://github.com/herumi/xbyak_riscv) (BSD-3-Clause)
and, for its tests only, [GoogleTest](https://github.com/google/googletest)
(BSD-3-Clause).

## Citing

Research that uses `rvjit` is asked to cite:

```bibtex
@inproceedings{santana2026rvjit,
  title     = {Just-in-Time Convolution and {GEMM} Code Generation for
               {SIMD} Architectures},
  author    = {de Limas Santana, Alexandre and Armejach, Adri{\`a} and
               Casas, Marc},
  booktitle = {2026 IEEE International Parallel and Distributed Processing
               Symposium (IPDPS)},
  pages     = {1105--1119},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/IPDPS65963.2026.00093}
}
```

## Author

Developed and maintained by Alexandre de Limas Santana in affiliation with
[Barcelona Supercomputing Center](https://www.bsc.es) (BSC).
