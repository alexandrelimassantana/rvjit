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

#ifndef RVJIT_TEST_BENCH_COMMON_CLI_HPP
#define RVJIT_TEST_BENCH_COMMON_CLI_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace rvjit_bench {

/// Splits a comma-separated flag value into its individual values
///
/// @details The cross-product primitive every multi-valued flag (e.g.
///     `--transA=0,1`) relies on: a driver reads each flag into its own
///     `std::vector`, then `for_each_combo` walks every combination
inline std::vector<std::string> split_csv(const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

/// Splits a `--name=value` (or bare `--name`) token into `name`/`value`
///
/// @note Returns false for a token with no `--` prefix (a positional spec)
inline bool parse_flag(
        const std::string &tok, std::string &name, std::string &value) {
    if (tok.size() < 2 || tok[0] != '-' || tok[1] != '-') return false;
    const auto eq = tok.find('=');
    if (eq == std::string::npos) {
        name = tok.substr(2);
        value.clear();
    } else {
        name = tok.substr(2, eq - 2);
        value = tok.substr(eq + 1);
    }
    return true;
}

inline bool to_bool(const std::string &s) {
    return s == "1" || s == "true";
}

inline std::vector<bool> parse_bool_list(const std::string &s) {
    std::vector<bool> out;
    for (auto &v : split_csv(s))
        out.push_back(to_bool(v));
    return out;
}

inline std::vector<int64_t> parse_int_list(const std::string &s) {
    std::vector<int64_t> out;
    for (auto &v : split_csv(s))
        out.push_back(std::atoll(v.c_str()));
    return out;
}

/// Extracts a batch-file-style `_n"name"` annotation from a positional spec
/// (e.g. `64x8x16_n"m_tiles"`), mutating `spec` to drop the annotation
///
/// @note Returns an empty string, leaving `spec` untouched, when absent
inline std::string extract_name(std::string &spec) {
    const auto pos = spec.find("_n\"");
    if (pos == std::string::npos) return {};
    const auto start = pos + 3;
    const auto end = spec.find('"', start);
    if (end == std::string::npos) return {};
    const std::string name = spec.substr(start, end - start);
    spec = spec.substr(0, pos);
    return name;
}

/// Splits an `A x B x C` positional dims spec (e.g. `64x8x16`) into integers
inline std::vector<int64_t> parse_dims(const std::string &spec) {
    std::vector<int64_t> out;
    std::string cur;
    for (char c : spec) {
        if (c == 'x' || c == 'X') {
            out.push_back(std::atoll(cur.c_str()));
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(std::atoll(cur.c_str()));
    return out;
}

/// `parse_dims` keeping each piece unevaluated (an `eval_vl_expr` expression)
inline std::vector<std::string> split_dims(const std::string &spec) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : spec) {
        if (c == 'x' || c == 'X') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
/// Evaluates `[N]v[+-N]` / `N`: `v` is `vl`, the elements one vector register
/// group holds at the case's LMUL, so a batch file's tile-relative sizes (e.g.
/// `2v+3` = two vector tiles and a tail) stay valid at every VLEN
///
/// @note Returns -1 on a malformed expression
inline int64_t eval_vl_expr(const std::string &e, int64_t vl) {
    size_t i = 0;
    const auto number = [&](int64_t &out) {
        if (i >= e.size() || e[i] < '0' || e[i] > '9') return false;
        out = 0;
        while (i < e.size() && e[i] >= '0' && e[i] <= '9')
            out = out * 10 + (e[i++] - '0');
        return true;
    };
    int64_t base = 0;
    if (i < e.size() && e[i] == 'v') {
        base = vl;
        ++i;
    } else if (number(base)) {
        if (i < e.size() && e[i] == 'v') {
            base *= vl;
            ++i;
        }
    } else {
        return -1;
    }
    if (i < e.size() && (e[i] == '+' || e[i] == '-')) {
        const bool neg = e[i++] == '-';
        int64_t off = 0;
        if (!number(off)) return -1;
        base += neg ? -off : off;
    }
    return i == e.size() ? base : -1;
}

/// Reads a batch file into a flat token list
///
/// @details Blank lines and anything from a `#` to end-of-line are dropped;
///     each remaining line is whitespace-split into tokens, the same shape
///     as a slice of `argv` -- so a batch file line is written exactly like
///     the CLI invocation it stands in for
inline std::vector<std::string> read_batch_file(const std::string &path) {
    std::vector<std::string> toks;
    std::ifstream f(path);
    if (!f) {
        std::fprintf(
                stderr, "error: cannot open batch file %s\n", path.c_str());
        return toks;
    }
    std::string line;
    while (std::getline(f, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream iss(line);
        std::string tok;
        while (iss >> tok)
            toks.push_back(tok);
    }
    return toks;
}

/// The directory component of `path`, or `"."` if `path` has none
inline std::string dirname_of(const std::string &path) {
    const auto pos = path.find_last_of('/');
    return pos == std::string::npos ? std::string(".") : path.substr(0, pos);
}

/// Recursively inlines every `--batch=<file>` token in `in` with that file's
/// own (recursively expanded) tokens, so a driver's own parser sees one flat
/// stream of tokens regardless of how many batch files nested to produce it
///
/// @details A relative `--batch=` value resolves against `base_dir` -- the
///     directory of the file that referenced it, not the process's current
///     directory -- so a batch file's own nested `--batch=harness_foo`
///     entries work regardless of where the driver is invoked from
inline std::vector<std::string> expand_tokens(
        const std::vector<std::string> &in, const std::string &base_dir = ".") {
    std::vector<std::string> out;
    for (auto &tok : in) {
        std::string name, value;
        if (parse_flag(tok, name, value) && name == "batch") {
            std::string path = value;
            if (!path.empty() && path[0] != '/') path = base_dir + "/" + path;
            auto sub = expand_tokens(read_batch_file(path), dirname_of(path));
            for (auto &t : sub)
                out.push_back(t);
        } else {
            out.push_back(tok);
        }
    }
    return out;
}

/// `expand_tokens` over `argv[start..argc)`, resolving top-level `--batch=`
/// values against the process's current directory
inline std::vector<std::string> expand_argv(
        int argc, char **argv, int start = 1) {
    std::vector<std::string> in(argv + start, argv + argc);
    return expand_tokens(in, ".");
}

/// Invokes `f(idx)` once per combination of `sizes[0] x sizes[1] x ...`
/// (`idx[d]` ranges over `[0, sizes[d])`) -- the cross-product every
/// multi-valued CLI flag combination expands a single case spec into
inline void for_each_combo(const std::vector<int> &sizes,
        const std::function<void(const std::vector<int> &)> &f) {
    for (auto s : sizes)
        if (s <= 0) return;
    std::vector<int> idx(sizes.size(), 0);
    while (true) {
        f(idx);
        int d = static_cast<int>(sizes.size()) - 1;
        while (d >= 0) {
            if (++idx[d] < sizes[d]) break;
            idx[d] = 0;
            --d;
        }
        if (d < 0) break;
    }
}

} // namespace rvjit_bench

#endif
