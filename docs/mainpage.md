# rvjit API reference

`rvjit` is a header-only C++11 library of reusable components for RV64 and
RVV JIT code generation, built on Xbyak_riscv. A kernel creates the components
through `rvjit::rvv::base_programming_model_t` and composes them.

| Component | Interface | RV64/RVV implementation |
|---|---|---|
| Constant arithmetic | `rvjit::operation::const_arithmetic_t` | `rvjit::rv64::const_arithmetic` |
| Arithmetic | `rvjit::operation::arithmetic_t` | `rvjit::rvv::arithmetic` |
| Memory | `rvjit::operation::memory_t` | `rvjit::rvv::memory` |
| Control flow | `rvjit::operation::control_flow_t` | `rvjit::rv64::control_flow` |
| Transfer | `rvjit::operation::transfer_t` | `rvjit::rv64::transfer` |
| Buffer pool | `rvjit::buffer::buffer_pool_t` | `rvjit::rvv::buffer` |
| Stack | `rvjit::stack_t` | `rvjit::rv64::stack_t` |
| Loops | `rvjit::loop::loopgen_t`, `rvjit::loop::plan_t` | `rvjit::rv64::loop` |
| Streams | `rvjit::stream::memory_operand::memory_operand_t` | `rvjit::rv64::stream::iterator` |
| Matrix multiplication | `rvjit::linalg::matmul::matmul_t` | `rvjit::rvv::linalg::matmul` |

The repository's README has usage examples, and `test/gtest/` and
`test/bench/` show each component in use.
