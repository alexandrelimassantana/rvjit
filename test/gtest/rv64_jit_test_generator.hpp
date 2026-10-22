/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef RVJIT_TEST_RV64_JIT_TEST_GENERATOR_HPP
#define RVJIT_TEST_RV64_JIT_TEST_GENERATOR_HPP

#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <string>
#include <utility>

#include <signal.h>
#include <unistd.h>

#include "xbyak_riscv/xbyak_riscv.hpp"
#include "gtest/gtest.h"

#include "rvjit/implementation/rvv/isa/buffer/architecture.hpp"
#include "rvjit/interface/isa/types.hpp"

#include "bench/common/asm_dump.hpp"

namespace rvjit {

namespace detail {

/// Directory to dump generated kernels' assembly into, or null if
/// `RVJIT_DUMP_DIR` is unset
inline const char *jit_dump_dir() {
    static const char *dir = std::getenv("RVJIT_DUMP_DIR");
    return dir;
}

/// Longjmp target for `run_with_timeout`'s SIGALRM handler
inline sigjmp_buf &timeout_jmpbuf() {
    static sigjmp_buf buf;
    return buf;
}

inline void timeout_handler(int) {
    siglongjmp(timeout_jmpbuf(), 1);
}

} // namespace detail

/// Runs `fn` under a SIGALRM watchdog
///
/// @param fn The call guarded
/// @param timeout_sec The seconds `fn` may run before the test fails
///
/// @note Fails the test instead of hanging when a `while_` or `loopgen_t`
///     branch never reaches its exit
template <typename Fn>
void run_with_timeout(Fn &&fn, unsigned timeout_sec = 5) {
    struct sigaction sa {};
    sa.sa_handler = detail::timeout_handler;
    sigemptyset(&sa.sa_mask);
    struct sigaction old_sa {};
    sigaction(SIGALRM, &sa, &old_sa);

    const bool timed_out = sigsetjmp(detail::timeout_jmpbuf(), 1) != 0;
    if (!timed_out) {
        alarm(timeout_sec);
        fn();
    }
    alarm(0);
    sigaction(SIGALRM, &old_sa, nullptr);

    if (timed_out) {
        FAIL() << "kernel call did not return within " << timeout_sec
               << "s -- possible infinite loop";
    }
}

/// Minimal JIT harness for rvjit component tests
///
/// @details Stands in for the production `jit_generator_t`:
///     - which needs `register_jit_code` and `get_max_cpu_isa`, hidden by
///       `-fvisibility=internal` in the shared library
///     - while the tests only JIT and run a probe, so this uses
///       `Xbyak_riscv::CodeGenerator` directly
class rv64_jit_test_generator_t : public Xbyak_riscv::CodeGenerator {
public:
    rv64_jit_test_generator_t() = default;

    /// @param maxCodeSize  Override the default 4KB code buffer; needed by
    ///     probe kernels that unroll a large number of iterations
    explicit rv64_jit_test_generator_t(size_t maxCodeSize)
        : Xbyak_riscv::CodeGenerator(maxCodeSize) {}

    /// @note When `RVJIT_DUMP_DIR` is set, also writes this kernel's code
    ///     as assembly to that directory, named after the current gtest
    ///     test (see `detail::dump_kernel_asm`)
    bool create_kernel() {
        generate();
        ready();
        if (const char *dir = detail::jit_dump_dir())
            dump_current_test_kernel_(dir);
        return getCode() != nullptr;
    }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t...);
        auto *fptr = reinterpret_cast<jit_kernel_func_t>(
                const_cast<uint8_t *>(getCode()));
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

protected:
    virtual void generate() = 0;

private:
    /// Names the dump after the current gtest test, deduplicated with a
    /// trailing counter so multiple `create_kernel()` calls within the
    /// same test (e.g. a loop over sub-cases) don't overwrite each other
    void dump_current_test_kernel_(const char *dir) const {
        std::string name = "kernel";
        if (const auto *info
                = ::testing::UnitTest::GetInstance()->current_test_info()) {
            name = std::string(info->test_suite_name()) + "." + info->name();
            for (char &c : name)
                if (c == '/') c = '_';
        }
        static std::map<std::string, int> counters;
        name += "." + std::to_string(counters[name]++);
        rvjit_bench::dump_kernel_asm(dir, name, getCode(), getSize());
    }
};

/// Concrete rv64_jit_test_generator_t whose generate() defers to a
/// caller-supplied callback, so scenario parameters are lambda-captured
/// locals instead of members of a subclass
class rv64_jit_probe_t : public rv64_jit_test_generator_t {
public:
    using generate_fn_t = std::function<void(rv64_jit_test_generator_t &)>;

    explicit rv64_jit_probe_t(generate_fn_t fn) : fn_(std::move(fn)) {}
    rv64_jit_probe_t(size_t maxCodeSize, generate_fn_t fn)
        : rv64_jit_test_generator_t(maxCodeSize), fn_(std::move(fn)) {}

protected:
    void generate() override { fn_(*this); }

private:
    generate_fn_t fn_;
};

/// Runs `k(args...)` under `run_with_timeout`
///
/// @param k The kernel called
/// @param args The arguments passed to the kernel
template <typename... Args>
void run_timed(const rv64_jit_test_generator_t &k, Args &&...args) {
    run_with_timeout([&] { k(std::forward<Args>(args)...); });
}

/// Builds the default architecture without the integer registers `excl`
///
/// @param excl The integer registers removed from the candidate list
inline rvv::buffer::architecture_t excluding_ints(
        const buffer::buffer_t &excl) {
    rvv::buffer::architecture_t source
            = rvv::buffer::architecture_t::default_list();
    source.integer.without(excl);
    return source;
}

/// Checks that `k` emitted `n` interleaved (probe, ref) instruction pairs
///
/// @param k The generator holding the code
/// @param n The number of pairs
///
/// @note Each pair must have identical encodings
inline void expect_matching_pairs(rv64_jit_test_generator_t &k, int n) {
    const auto *code = k.getCode<const uint32_t *>();
    ASSERT_GT(k.getSize(), n * 2 * sizeof(uint32_t));
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(code[2 * i], code[2 * i + 1]) << "mismatch at pair " << i;
}

} // namespace rvjit

#endif
