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

#ifndef RVJIT_TEST_BENCH_COMMON_DTYPE_HPP
#define RVJIT_TEST_BENCH_COMMON_DTYPE_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "rvjit/interface/isa/operand.hpp"

namespace rvjit_bench {

using dtype_t = rvjit::type_t;

/// The dtypes a bench driver can name on its command line
inline const char *dtype_name(dtype_t dt) {
    switch (dt) {
        case dtype_t::s8: return "s8";
        case dtype_t::u8: return "u8";
        case dtype_t::s16: return "s16";
        case dtype_t::u16: return "u16";
        case dtype_t::s32: return "s32";
        case dtype_t::s64: return "s64";
        case dtype_t::f16: return "f16";
        case dtype_t::bf16: return "bf16";
        case dtype_t::f32: return "f32";
        case dtype_t::f64: return "f64";
        default: return "?";
    }
}

inline bool parse_dtype(const std::string &s, dtype_t &dt) {
    for (dtype_t d : {dtype_t::s8, dtype_t::u8, dtype_t::s16, dtype_t::u16,
                 dtype_t::s32, dtype_t::s64, dtype_t::f16, dtype_t::bf16,
                 dtype_t::f32, dtype_t::f64}) {
        if (s == dtype_name(d)) {
            dt = d;
            return true;
        }
    }
    return false;
}

inline bool dtype_is_float(dtype_t dt) {
    return dt == dtype_t::f16 || dt == dtype_t::bf16 || dt == dtype_t::f32
            || dt == dtype_t::f64;
}

inline bool dtype_is_signed(dtype_t dt) {
    return dt != dtype_t::u8 && dt != dtype_t::u16;
}

inline int dtype_bytes(dtype_t dt) {
    return rvjit::sewb(dt);
}

/// Unit roundoff of a float dtype
inline double dtype_eps(dtype_t dt) {
    switch (dt) {
        case dtype_t::f16: return 4.8828125e-4;
        case dtype_t::bf16: return 3.90625e-3;
        case dtype_t::f32: return 1.1920929e-7;
        case dtype_t::f64: return 2.220446049250313e-16;
        default: return 0.0;
    }
}

/// Largest finite value of an integer dtype
inline int64_t dtype_int_max(dtype_t dt) {
    switch (dt) {
        case dtype_t::s8: return 127;
        case dtype_t::u8: return 255;
        case dtype_t::s16: return 32767;
        case dtype_t::u16: return 65535;
        case dtype_t::s32: return 2147483647;
        default: return 9223372036854775807LL;
    }
}

inline uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu)
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        const uint32_t m = mant | 0x800000u;
        const int shift = 14 - exp;
        uint32_t half = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
}

inline float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            int e = -1;
            uint32_t m = mant;
            do {
                ++e;
                m <<= 1;
            } while (!(m & 0x400u));
            x = sign | (static_cast<uint32_t>(127 - 15 - e) << 23)
                    | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    x += 0x7fffu + ((x >> 16) & 1u);
    return static_cast<uint16_t>(x >> 16);
}

inline float bf16_bits_to_f32(uint16_t b) {
    const uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

/// Wraps `v` to the two's complement range of an integer dtype
inline int64_t wrap_int(int64_t v, dtype_t dt) {
    switch (dt) {
        case dtype_t::s8: return static_cast<int8_t>(v);
        case dtype_t::u8: return static_cast<uint8_t>(v);
        case dtype_t::s16: return static_cast<int16_t>(v);
        case dtype_t::u16: return static_cast<uint16_t>(v);
        case dtype_t::s32: return static_cast<int32_t>(v);
        default: return v;
    }
}

/// Reads element `i` of a buffer of `dt` elements as a double
inline double load_elem(const uint8_t *base, int64_t i, dtype_t dt) {
    const uint8_t *p = base + i * dtype_bytes(dt);
    switch (dt) {
        case dtype_t::s8: return static_cast<int8_t>(*p);
        case dtype_t::u8: return *p;
        case dtype_t::s16: {
            int16_t v;
            std::memcpy(&v, p, 2);
            return v;
        }
        case dtype_t::u16: {
            uint16_t v;
            std::memcpy(&v, p, 2);
            return v;
        }
        case dtype_t::s32: {
            int32_t v;
            std::memcpy(&v, p, 4);
            return v;
        }
        case dtype_t::s64: {
            int64_t v;
            std::memcpy(&v, p, 8);
            return static_cast<double>(v);
        }
        case dtype_t::f16: {
            uint16_t v;
            std::memcpy(&v, p, 2);
            return f16_bits_to_f32(v);
        }
        case dtype_t::bf16: {
            uint16_t v;
            std::memcpy(&v, p, 2);
            return bf16_bits_to_f32(v);
        }
        case dtype_t::f32: {
            float v;
            std::memcpy(&v, p, 4);
            return v;
        }
        default: {
            double v;
            std::memcpy(&v, p, 8);
            return v;
        }
    }
}

/// Writes `v`, rounded (floats) or wrapped (integers), as element `i`
inline void store_elem(uint8_t *base, int64_t i, dtype_t dt, double v) {
    uint8_t *p = base + i * dtype_bytes(dt);
    if (!dtype_is_float(dt)) {
        const int64_t w = wrap_int(static_cast<int64_t>(std::llround(v)), dt);
        std::memcpy(p, &w, dtype_bytes(dt));
        return;
    }
    switch (dt) {
        case dtype_t::f16: {
            const uint16_t b = f32_to_f16_bits(static_cast<float>(v));
            std::memcpy(p, &b, 2);
            break;
        }
        case dtype_t::bf16: {
            const uint16_t b = f32_to_bf16_bits(static_cast<float>(v));
            std::memcpy(p, &b, 2);
            break;
        }
        case dtype_t::f32: {
            const float f = static_cast<float>(v);
            std::memcpy(p, &f, 4);
            break;
        }
        default: std::memcpy(p, &v, 8); break;
    }
}

} // namespace rvjit_bench

#endif
