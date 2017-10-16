
#include "utilities.hpp"
#include "system.hpp"

#include "debug.hpp"

namespace jup {

static Arena tmp_arena;

// see header
void* tmp_alloc(int size) {
    return tmp_arena.allocate(size);
}

// see header
void tmp_alloc_reset() {
    tmp_arena.reset();
}

// see header
Arena& tmp_alloc_arena() {
    return tmp_arena;
}

static Buffer tmp_buffer;

// see header
Buffer& tmp_alloc_buffer() {
    return tmp_buffer;
}

jup_str nice_bytes(u64 bytes) {
    if (bytes < 1000) {
        return jup_printf("%d B", bytes);
    } else if ((bytes >> 10) < 1000) {
        return jup_printf("%.2f KiB", (float)bytes / 1024.f);
    } else if ((bytes >> 20) < 1000) {
        return jup_printf("%.2f MiB", (float)(bytes >> 10) / 1024.f);
    } else if ((bytes >> 30) < 1000) {
        return jup_printf("%.2f GiB", (float)(bytes >> 20) / 1024.f);
    } else {
        return jup_printf("%.2f TiB", (float)(bytes >> 30) / 1024.f);
    }
}

jup_str nice_time(double s) {
    if (s < 100e-9) {
        return jup_printf("%.0fns", s * 1e9);
    } else if (s < 100e-6) {
        return jup_printf("%.2fus", s * 1e6);
    } else if (s < 100e-3) {
        return jup_printf("%.2fms", s * 1e3);
    } else if (s < 60) {
        return jup_printf("%.2fs", s);
    } else if (s < 3600*24) {
        double h = std::floor(s / 3600.0);
        double m = std::floor((s - h*3600) / 60.0);
        return jup_printf("%02.0f:%02.0f:%02.0f", h, m, s - h*3600 - m*60);
    } else {
        double y = std::floor(s / 3600.0 / 24.0);
        double h = std::floor((s - y*3600.0*24.0) / 3600.0);
        double m = std::floor((s - y*3600.0*24.0 - h*3600.0) / 60.0);
        return jup_printf("%.0f:%02.0f:%02.0f:%02.0f", y, h, m, s - y*3600*24 - h*3600 - m*60);
    }
}

jup_str nice_oct(Buffer_view data, bool swap)  {
    char* tmp = (char*)tmp_alloc((data.size() * 8 + 2) / 3 + 1);

    auto get_byte = [swap, data](int i) {
        return (u8)data[swap ? data.size() -i-1 : i];
    };
    auto get_bits = [&get_byte](int i) {
        u8 val = get_byte(i/8);
        if (i%8 < 6) {
            val >>= (5 - i%8);
        } else {
            u8 oth = get_byte(i/8 + 1);
            val = (val << (i%8 - 5)) | (oth >> (13 - i%8));
        }
        return val & 7;
    };

    int i_bit = (data.size() * 8) % 3;
    int i = 0;
    bool flag = false;
    if (i_bit > 0) {
        u8 val = get_byte(0) >> (8 - i_bit);
        if (val or flag) {
            tmp[i++] = '0' + val;
            flag = true;
        }
    }
    
    for (;i_bit < data.size() * 8; i_bit += 3) {
        u8 val = get_bits(i_bit);
        if (val or flag) {
            tmp[i++] = '0' + val;
            flag = true;
        }
    }

    tmp[i] = '\0';
    return {tmp, i};
}

jup_str nice_hex(Buffer_view data, bool swap) {
    char* tmp = (char*)tmp_alloc(data.size() * 2 + 1);

    for (int i = 0; i < data.size(); ++i) {
        char c1 = (u8)data[i] >> 4;
        char c2 = (u8)data[i] & 15;
        c1 = c1 < 10 ? c1 + '0' : c1 - 10 + 'a';
        c2 = c2 < 10 ? c2 + '0' : c2 - 10 + 'a';
        tmp[2*i]     = c1;
        tmp[2*i + 1] = c2;
    }
    tmp[data.size() * 2] = '\0';
    if (swap) {
        for (int i = 0; 2*i < data.size(); ++i) {
            std::swap(tmp[2*i],   tmp[2*(data.size()-1 - i)]    );
            std::swap(tmp[2*i+1], tmp[2*(data.size()-1 - i) + 1]);
        }
    }
    return {tmp, data.size() * 2};
}

void print_wrapped(std::ostream& out, jup_str str) {
    int width = get_terminal_width();

    if (width < 40) width = 40;

    int i = 0;
    while (i < str.size()) {
        int start = i;
        while (str[i] == ' ' and i < str.size()) ++i;
        if (i == str.size()) break;
            
        int indent = i - start;
        while (true) {
            start = i;
            while (str[i] != '\n' and i < str.size() and indent + i - start < width) ++i;

            if (i == str.size()) {
                break;
            } else if (str[i] == '\n') {
                for (int j = 0; j < indent; ++j) out.put(' ');
                out.write(str.data() + start, ++i - start);
                break;
            }
            
            while (i > start and str[i] != ' ') --i;
            if (i == start) {
                i += width - indent;
                for (int j = 0; j < indent; ++j) out.put(' ');
                out.write(str.data() + start, i - start);
            } else {
                for (int j = 0; j < indent; ++j) out.put(' ');
                out.write(str.data() + start, i++ - start);
                out.put('\n');
            }
        }
    }
}

class Dummy_streambuf: public std::streambuf {
public:
    std::streamsize xsputn (char const* str, std::streamsize n) override {
        return n;
    }
    int overflow (int c) override {
        return 1;
    }
};

class Dummy_ostream: public std::ostream {
public:
    Dummy_ostream(): std::ostream (&buffer) {}
private:
    Dummy_streambuf buffer;
};

static Dummy_ostream jnull_stream;
std::ostream& jnull = jnull_stream;


jup_str jup_err_messages[] = {
    /* 0 */ nullptr,
    /* 1 */ "String is empty",
    /* 2 */ "Invalid character",
    /* 3 */ "Out of range (too low)",
    /* 4 */ "Out of range (too high)",
    /* 5 */ "Unexpected end of input"
    /* 6 */ "Value too close to zero"
    /* 7 */ "Extra characters"
    /* 8 */ "Expected an integer"
};

struct Number_sci {
    enum Type: u8 {
        NORMAL, T_INFINITY, T_NAN
    };
    
    u8 type;
    bool sign;
    u64 m; // mantissa
    int e; // exponent
};

__jup_dbg(Number_sci, type, sign, m, e)

/**
 * Converts a string into a number. This function returns imprecise results!
 */
static u16 jup_sto_helper(jup_str str, Number_sci* into, u8 flags = 0) {
    assert(into);
    if (not str) return 1;

    bool sign = false;
    int i = 0;
    while (i < str.size() and (str[i] == '-' or str[i] == '+')) {
        sign ^= str[i] == '-';
        ++i;
    }
    if (i == str.size()) return 5;

    auto cmp_ci = [str, &i](char const* s) {
        if (i + (int)std::strlen(s) > str.size()) return false;
        for (int j = 0; j < (int)std::strlen(s); ++j) {
            if (str[i + j] != s[j] and str[i+j] != s[j] + 'A' - 'a') return false;
        }
        i += std::strlen(s);
        return true;
    };

    if (flags & jup_sto::ALLOW_INFINITY) {
        if (cmp_ci("infty") or cmp_ci("infinity") or cmp_ci("inf")) {
            if (i < str.size()) return 7;
            *into = {Number_sci::T_INFINITY, sign, 0, 0};
            return 0;
        }
    } else if (flags & jup_sto::ALLOW_NAN) {
        if (cmp_ci("nan")) {
            if (i < str.size()) return 7;
            *into = {Number_sci::T_NAN, sign, 0, 0};
            return 0;
        }
    }
    
    u64 base = 10;
    if (str[i] == '0' and i + 1 < str.size()) {
        ++i;
        if (str[i] == 'x' or str[i] == 'X') {
            base = 16; ++i;
        } else if (str[i] == 'b' or str[i] == 'B') {
            base = 2; ++i;
        } else if ('0' <= str[i] and str[i] <= '9') {
            base = 8;
        } else {
            // nothing
        }
    }
    if (i == str.size()) return 5;

    u64 m = 0;
    bool overflow = false;
    bool do_exp = false;
    bool do_frac = false;
    for (; i < str.size(); ++i) {
        char c = str[i];
        u64 val = 0;
        if ('0' <= c and c <= '9') {
            val = c - '0';
        } else if (base == 16 and 'a' <= c and c <= 'z') {
            val = c - 'a';
        } else if (base == 16 and 'A' <= c and c <= 'Z') {
            val = c - 'A';
        } else if (base == 10 and (c == 'e' or c == 'E')) {
            do_exp = true; ++i; break;
        } else if (c == '.') {
            do_frac = true; ++i; break;
        } else {
            return 2;
        }
        if (val >= base) { return 2; }

        if (__builtin_mul_overflow(m, base, &m)) {
            overflow = true; break;
        } else if (__builtin_add_overflow(m, val, &m)) {
            overflow = true; break;
        }
    }
    if (overflow) return sign ? 3 : 4;

    u64 frac = 0;
    u64 frac_exp = 0;
    if (do_frac) {
        for (; i < str.size(); ++i) {
            char c = str[i];
            u64 val = 0;
            if ('0' <= c and c <= '9') {
                val = c - '0';
            } else if (base == 16 and 'a' <= c and c <= 'z') {
                val = c - 'a';
            } else if (base == 16 and 'A' <= c and c <= 'Z') {
                val = c - 'A';
            } else if (base == 10 and (c == 'e' or c == 'E')) {
                do_exp = true; ++i; break;
            } else {
                return 2;
            }
            if (val >= base) { return 2; }

            if (__builtin_mul_overflow(frac, base, &frac)) {
                overflow = true; break;
            } else if (__builtin_add_overflow(frac, val, &frac)) {
                overflow = true; break;
            } else if (__builtin_add_overflow(frac_exp, 1, &frac_exp)) {
                overflow = true; break;
            }
        }
    }
    if (overflow) {
        // Skip the rest of the factional part, loose the precision
        while (i < str.size() and '0' <= str[i] and str[i] <= '9') ++i;
        overflow = false;
    }
    
    int exp = 0;
    if (do_exp) {
        bool exp_sign = false;
        u64 exp_val = 0;
        while (i < str.size() and (str[i] == '-' or str[i] == '+')) {
            exp_sign ^= str[i] == '-';
            ++i;
        }
        if (i == str.size()) return 5;
    
        for (; i < str.size(); ++i) {
            char c = str[i];
            u64 val = 0;
            if ('0' <= c and c <= '9') {
                val = c - '0';
            } else {
                return 2;
            }

            if (__builtin_mul_overflow(exp_val, 10, &exp_val)) {
                overflow = true; break;
            } else if (__builtin_add_overflow(exp_val, val, &exp_val)) {
                overflow = true; break;
            }
        }
        
        if (__builtin_mul_overflow(exp_val, exp_sign ? -1 : 1, &exp)) {
            overflow = true;
        }

        if (overflow) return exp_sign ? (sign ? 3 : 4) : 6;
    }

    // Add the fractional part
    for (u64 i = 0; i < frac_exp; ++i) {
        if (__builtin_mul_overflow(m, base, &m)) {
            overflow = true; break;
        }
    }
    if (overflow) return sign ? 3 : 4;
    if (__builtin_add_overflow(m, frac, &m)) {
        overflow = true;
    } else if (__builtin_sub_overflow(exp, frac_exp, &exp)) {
        overflow = true;
    }
    if (overflow) return sign ? 3 : 4;

    // Convert exponent into base 2
    // TODO: Implement correct rounding
    int exp_;
    if (m == 0 or exp == 0) {
        exp_ = 0;
    } else if (exp > 0 and std::log2(base) * (double)exp < __builtin_clzll(m)) {
        // If the number is an 64-bit integer, represent it directly
        for (int i = 0; i < exp; ++i) {
            m *= 10;
        }
        exp_ = 0;
    } else if (base == 10) {
        u64 shift = __builtin_clzll(m);
        m <<= shift;
        if (exp > (int)((double)std::numeric_limits<int>::max() / std::log2(base))
            or exp < (int)((double)std::numeric_limits<int>::min() / std::log2(base))) {
            return exp < 0 ? 6 : (sign ? 3: 4);
        }
        exp_ = (int)(std::ceil(exp * std::log2((long double)base)));

        long double m_ld = (long double)m;
        if (exp > 0) {
            long double d = 10.l;
            u64 i = (u64)exp;
            while (i) {
                if (i & 1) m_ld *= d;
                d *= d;
                i >>= 1;
            }
        } else {
            long double d = 10.l;
            u64 i = (u64)-exp;
            while (i) {
                if (i & 1) m_ld /= d;
                d *= d;
                i >>= 1;
            }
        }

        m = (u64)(std::ldexp(m_ld, -exp_));
        exp_ -= shift;
    
    } else {
        assert(exp == 0);
        exp_ = 0;
    }

    if (flags & jup_sto::_MAX_LEFT) {
        if (m != 0) {
            exp_ -= __builtin_clzll(m);
            m <<= __builtin_clzll(m);
        }
    } else {
        if (m != 0) {
            exp_ += __builtin_ctzll(m);
            m >>= __builtin_ctzll(m);
        }
    }
    
    if (flags & jup_sto::_ONLY_INTEGER) {
        if (exp_ != 0) {
            assert(m != 0);
            if (exp_ > 0 and __builtin_clzll(m) >= exp_) {
                m <<= exp_;
                exp_ = 0;
            } else {
                return 8;
            }
        }
    }

    *into = {Number_sci::NORMAL, sign, m, exp_};
    return 0;
}

template <typename T>
u16 jup_stox_helper_int(jup_str str, T* into, u8 flags) {
    static_assert(sizeof(T) <= sizeof(u64) and std::is_integral<T>::value);
    assert(into);
    assert(flags == 0);
    
    Number_sci n;
    if (auto code = jup_sto_helper(str, &n, flags | jup_sto::_ONLY_INTEGER)) {
        return code;
    }

    assert(n.e == 0); // due to the _ONLY_INTEGER flag
    assert(n.type == Number_sci::NORMAL);

    if (std::is_unsigned<T>::value and n.sign and n.m) {
        return 3;
    }
    if (n.m > (u64)std::numeric_limits<T>::max() + n.sign) {
        return n.sign ? 3 : 4;
    }

    *into = n.sign ? (T)-n.m : (T)n.m;
    return 0;
}

u16 jup_stox(jup_str str, u8*  into, u8 flags) { return jup_stox_helper_int(str, into, flags); }
u16 jup_stox(jup_str str, s8*  into, u8 flags) { return jup_stox_helper_int(str, into, flags); }
u16 jup_stox(jup_str str, u16* into, u8 flags) { return jup_stox_helper_int(str, into, flags); }
u16 jup_stox(jup_str str, s16* into, u8 flags) { return jup_stox_helper_int(str, into, flags); }
u16 jup_stox(jup_str str, u32* into, u8 flags) { return jup_stox_helper_int(str, into, flags); }
u16 jup_stox(jup_str str, s32* into, u8 flags) { return jup_stox_helper_int(str, into, flags); }
u16 jup_stox(jup_str str, u64* into, u8 flags) { return jup_stox_helper_int(str, into, flags); }
u16 jup_stox(jup_str str, s64* into, u8 flags) { return jup_stox_helper_int(str, into, flags); }

u16 jup_stox(jup_str str, float* into, u8 flags) {
    static_assert(std::numeric_limits<float>::is_iec559);
    assert(into);

    Number_sci n;
    if (auto code = jup_sto_helper(str, &n, flags | jup_sto::_MAX_LEFT)) {
        return code;
    }
    
    // We interpret n.m as a real in [0, 2)
    n.e += 63;
    assert(n.m == 0 or (n.m & (1ull << 63)));

    union {
        u32 d = 0;
        float result;
    };

    // sign
    d ^= ((u32)n.sign << 31);
    
    if (n.type == Number_sci::T_NAN) {
        d = 0x7fc00000ull;
    } else if (n.type == Number_sci::T_INFINITY) {
        d |= 0x7f800000ull;
    } else if (n.type == Number_sci::NORMAL)  {
        // Take care of the normal-denormal cutoff
        if (n.e == -127) {
            if (n.m >= 0xffffff0000000000ull) {
                n.m = 1ull << 63;
                n.e += 1;
            }
        } else if (n.e <= -150) {
            if (n.e == -150 and n.m > (1ull << 63)) {
                n.m = 1ull << 63;
                n.e += 1;
            } else {
                n.m = 0;
                n.e = 0;
            }
        }
        
        if (n.m == 0) {
            // nothing, mantissa and exponent are already zero
        } else if ((n.e > -127 and n.e < 127) or (n.e == 127 and n.m < 0xffffff7000000000ull)) {
            // normalized
            u64 m_ = n.m >> 40;
            s64 exp_ = n.e;
            u64 round = n.m & 0xffffffffffull;

            if (not (round & 0x8000000000ull)) {
                // round down
            } else if (round & 0x7fffffffffull) {
                // round up
                m_ += 1;
            } else {
                assert(round == 0x8000000000ull);
                // round towards even
                m_ += m_ & 1;
            }
            if (m_ & (1ull << 25)) {
                m_ >>= 1;
                exp_ += 1;
            }

            assert(m_ < (1ull << 25) and exp_ >= -126 and exp_ <= 1027);
            d |= (m_ & ~(1ull << 23));
            d |= (u64)(exp_ + 127) << 23;
        } else if (n.e >= -149 and n.e <= -127) {
            // denormalized
            u64 shift = 41 - (127 + n.e);
            u64 m_ = n.m >> shift;
            u64 round = (n.m >> (shift - 41)) & 0xffffffffffull;

            if (not (round & 0x8000000000ull)) {
                // round down
            } else if (round & 0x7fffffffffull) {
                // round up
                m_ += 1;
            } else {
                assert(round == 0x8000000000ull);
                // round towards even
                m_ += m_ & 1;
            }
            
            assert(m_ < (1ull << 24));
            d |= m_;
            // exponent already 0
        } else {
            return n.e < 0 ? 6 : (n.sign ? 3 : 4);
        }
    } else {
        assert(false);
    }

    *into = result;
    return 0;
}

u16 jup_stox(jup_str str, double* into, u8 flags) {
    static_assert(std::numeric_limits<double>::is_iec559);
    assert(into);

    Number_sci n;
    if (auto code = jup_sto_helper(str, &n, flags | jup_sto::_MAX_LEFT)) {
        return code;
    }
    
    // We interpret n.m as a real in [0, 2)
    n.e += 63;
    assert(n.m == 0 or (n.m & (1ull << 63)));

    union {
        u64 d = 0;
        double result;
    };

    // sign
    d ^= ((u64)n.sign << 63);
    
    if (n.type == Number_sci::T_NAN) {
        d = 0x7ff8000000000000ull;
    } else if (n.type == Number_sci::T_INFINITY) {
        d |= 0x7ff0000000000000ull;
    } else if (n.type == Number_sci::NORMAL)  {
        // Take care of the normal-denormal cutoff
        if (n.e == -1023) {
            if (n.m >= 0xfffffffffffff800ull) {
                n.m = 1ull << 63;
                n.e += 1;
            }
        } else if (n.e <= -1075) {
            if (n.e == -1075 and n.m > (1ull << 63)) {
                n.m = 1ull << 63;
                n.e += 1;
            } else {
                n.m = 0;
                n.e = 0;
            }
        }
        
        if (n.m == 0) {
            // nothing, mantissa and exponent are already zero
        } else if ((n.e > -1023 and n.e < 1023) or (n.e == 1023 and n.m < 0xfffffffffffffc00ull)) {
            // normalized
            u64 m_ = n.m >> 11;
            s64 exp_ = n.e;
            u64 round = n.m & 0x7ffull;

            if (not (round & 0x400)) {
                // round down
            } else if (round & 0x3ff) {
                // round up
                m_ += 1;
            } else {
                assert(round == 0x400);
                // round towards even
                m_ += m_ & 1;
            }
            if (m_ & (1ull << 54)) {
                m_ >>= 1;
                exp_ += 1;
            }

            assert(m_ < (1ull << 54) and exp_ >= -1022 and exp_ <= 1023);
            d |= (m_ & ~(1ull << 52));
            d |= (u64)(exp_ + 1023) << 52;
        } else if (n.e >= -1074 and n.e <= -1023) {
            // denormalized
            u64 shift = 12 - (1023 + n.e);
            u64 m_ = n.m >> shift;
            u64 round = (n.m >> (shift - 12)) & 0xfffull;

            if (not (round & 0x800)) {
                // round down
            } else if (round & 0x7ff) {
                // round up
                m_ += 1;
            } else {
                assert(round == 0x800);
                // round towards even
                m_ += m_ & 1;
            }

            assert(m_ < (1ull << 53));
            d |= m_;
            // exponent already 0
        } else {
            return n.e < 0 ? 6 : (n.sign ? 3 : 4);
        }
    } else {
        assert(false);
    }

    *into = result;
    return 0;
}

// from https://en.wikipedia.org/wiki/Xorshift
u64 Rng::rand() {
    u64 x = rand_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rand_state = x;
    return x * 0x2545f4914f6cdd1dull;
}

bool Rng::gen_bool(u8 perbyte) {
    return (rand() & 0xff) < perbyte;
}

u64 Rng::gen_uni(u64 max) {
    u64 x = rand();
    return x % max;
}

u8 Rng::gen_exp(u8 perbyte) {
    u64 x = rand();
    u64 y = std::numeric_limits<u64>::max();
    y = (y >> 8) * perbyte;

    u8 i = 0;
    while (x < y) {
        i += x > y;
        y = (y * perbyte) >> 8;
    }
    return i;
}

float Rng::gen_any_float() {
    union {
        u32 d;
        float result;
    };
    d = rand();
    if (((d >> 23) & 0xff) == 0xff) {
        d &= ~(0xffull << 23);
        d |= gen_uni(0xff) << 23;
    }
    return result;
}

double Rng::gen_any_double() {
    union {
        u64 d;
        double result;
    };
    d = rand();
    if (((d >> 52) & 0x7ff) == 0x7ff) {
        d &= ~(0x7ffull << 52);
        d |= gen_uni(0x7ff) << 52;
    }
    return result;
}

Rng global_rng;

Timer::Timer(u64 progress_target_):
    progress_target{progress_target_},
    last_bytes{0},
    cur_duration{0}
{
    start_time = elapsed_time();
    next_update = start_time + update_delay;
}

bool Timer::update() {
    double t = elapsed_time();
    if (t > next_update) {
        cur_duration = t - next_update + update_delay;
        next_update = t + update_delay;
        return true;
    } else {
        return false;
    }
}

jup_str Timer::progress(u64 have) {
    return jup_printf("%5.2f%%", (float)have / (float)progress_target * 100.f);
}

jup_str Timer::bytes(u64 have) {
    u64 diff = have - last_bytes;
    last_bytes = have;
    return jup_printf("%s/s", nice_bytes((u64)(diff / cur_duration)));
}

jup_str Timer::total() {
    return nice_time(elapsed_time() - start_time);
}


Histogram::Histogram(int size_) {
    n = 0;
    b = size_;
    _data = make_unique_void((b + 1) * (sizeof(float) + sizeof(int)));
    q_ = {(float*)_data.get(), b+1};
    n_ = {(int*)q_.end(), b+1};
}

void Histogram::add(float x) {
    // p-squared algorithm for quantile approximation
    if (n < b+1) {
        q_[n] = x;
        n_[n] = n + 1;
        ++n;
        if (n == b+1) {
            std::sort(q_.begin(), q_.end());
        }
    } else {
        ++n;
        int k = -1;
        for (float y: q_) k += y <= x;
        if (k == -1) {
            q_[0] = x;
            ++k;
        } else if (k == b) {
            q_[k] = x;
            --k;
        }
        for (int i = k+1; i < b+1; ++i) {
            ++n_[i];
        }
        for (int i = 1; i < b; ++i) {
            float nn = 1.f + (float)(i * (n-1)) / (float)b;
            float di = nn - (float)n_[i];
            if ((di >= 1.f and n_[i+1] - n_[i] > 1) or (di <= -1.f and n_[i-1] - n_[i] < -1)) {
                di = std::copysign(1.f, di);
                float qi = q_[i] + di/(float)(n_[i+1]-n_[i-1]) * ( (n_[i]-n_[i-1]+di) * (q_[i+1]-q_[i])
                    / (float)(n_[i+1]-n_[i]) + (n_[i+1]-n_[i]-di) * (q_[i]-q_[i-1]) / (float)(n_[i]-n_[i-1]) );
                if (q_[i-1] < qi and qi < q_[i+1]) {
                    q_[i] = qi;
                } else {
                    q_[i] = q_[i] + di * (q_[i+(int)di] - q_[i]) / (float)(n_[i+(int)di] - n_[i]);
                }
                n_[i] += (int)di;
            }
        }
    }    
}

void Histogram::print(int max_width, int max_height) const {
    if (max_width == -1) {
        max_width = get_terminal_width();
    }
    if (max_height == -1) {
        max_height = 20;
    }
    int width = max_width - 4;
    int height = max_height - 3;
    assert(width > 0 and height > 0);
    if (n < b+1) {
        jerr << "Error: " << "Tried to print a histogram without enough data (got " << n
             << ", need " << b+1 << "\n";
        die();
    }

    float x0 = q_[0];
    float x1 = q_[b];
    float xs = (x1 - x0) / (float)width;
    
    auto vals = __jup_stack_array(float, width);
    std::memset(vals.data(), 0, vals.as_bytes().size());
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < width; ++j) {
            float l = std::max(q_[i], x0 + xs*j);
            float r = std::min(q_[i+1], x0 + xs*(j + 1));
            float frac;
            if (std::abs(q_[i] - q_[i+1]) < 1e-6 * (x1 - x0) and l <= r) {
                frac = 1.f;
            } else {
                frac = std::max((r - l) / (q_[i+1] - q_[i]), 0.f);
            }
            //vals[j] += frac * (n_[i+1] - n_[i]);
            vals[j] += frac;
        }
    }

    float val_max = *std::max_element(vals.begin(), vals.end());
    
    jout << "Histogram (n = " << n << ", b = " << b << "):" << endl;
    for (int i = height - 1; i >= 0; --i) {
        jout << "  ";
        for (int j = 0; j < width; ++j) {
            int y = (int)std::round(vals[j] / val_max * height);
            jout.put(y > i ? '#' : ' ');
        }
        jout << "  \n";
    }
    jout << "  ";
    for (int i = 0; i < width; ++i) jout.put('-');
    jout.put('\n');
    auto str1 = jup_printf("%.2le", (double)x0);
    auto str2 = jup_printf("%.2le", (double)x1);
    jout << "  " << str1;
    for (int i = 0; i < width - str1.size() - str2.size(); ++i) jout.put(' ');
    jout << str2;
    jout << endl;
}

/*
void histogram_test() {
    Histogram h {100};
    std::mt19937_64 mt;
    std::normal_distribution<float> dist;
    for (int i = 0; i < 1000000; ++i) {
        h.add(dist(mt));
    }

    auto cdf = [](float x) {
        return 0.5f * std::erfc(-x * (float)M_SQRT1_2);
    };
    for (int i = 0; i < h.b + 1; ++i) {
        jout << jup_printf("%2d %9.2e\n", i, (double)(cdf(h.q_[i]) - (float)i/(float)h.b));
    }
    h.print();
}
*/
    


} /* end of namespace jup */
