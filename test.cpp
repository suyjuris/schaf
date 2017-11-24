
#include "test.hpp"

#include "utilities.hpp"

namespace jup {

template <typename real_t, typename repr_t, int max_exp>
void test_conversions_helper(const char* fmt) {
    Rng rng;
    int warn_count = 0;
    int warn_total = 0;
    
    auto chk1 = [&warn_count, &warn_total, fmt](jup_str str, real_t val, bool allow_obo = false, u8 flags = 0) {
        union { real_t val2; repr_t val_repr; };
        val2 = val;
        union { real_t i = 0; repr_t i_repr; };
        
        auto code = jup_stox(str, &i, flags);
        if (code != 0) {
            jerr << "Error: code " << code << " while parsing " << str << "\n";
            die();
        }
        warn_total += allow_obo;
        if (i_repr != val_repr) {
            if (i_repr - val_repr + 1 <= 2 and allow_obo) {
                ++warn_count;
            } else {
                jerr << "Error: wrong parsing on " << str << ", expected " << jup_printf(fmt, val) << ", got "
                << jup_printf(fmt, i) << " (relative error "
                << jup_printf(fmt, std::abs(i - val) / val) << ", bit-error "
                << nice_hex(i_repr ^ val_repr) << ")\n";
                die();
            }
        }
    };
    auto chk2 = [&chk1, fmt](real_t from, real_t to, int num) {
        for (int i = 0; i < num; ++i) {
            chk1(jup_printf(fmt, from), from);
            from = std::nextafter(from, to);
        }
    };
    auto chk3 = [&chk1, &rng, fmt](int num) {
        for (int i = 0; i < num; ++i) {
            real_t d = rng.gen_any<real_t>();
            chk1(jup_printf(fmt, d), d);
        }
    };
    auto chk4 = [&chk1, &rng](int num) {
        for (int j = 0; j < num; ++j) {
            auto s = jup_printf("%" PRIu32 ".%" PRIu32 "e%" PRId64, (u32)rng.rand() >> 2,
                (u32)rng.rand() >> 2, (s64)rng.gen_uni(max_exp*2)-max_exp);
            chk1(s, std::strtod(s.c_str(), nullptr), true);
        }
    };
    
    chk1("inf", std::numeric_limits<real_t>::infinity(), false, jup_sto::ALLOW_INFINITY);
    chk1("inF", std::numeric_limits<real_t>::infinity(), false, jup_sto::ALLOW_INFINITY);
    chk1("infinity", std::numeric_limits<real_t>::infinity(), false, jup_sto::ALLOW_INFINITY);
    chk1("infty", std::numeric_limits<real_t>::infinity(), false, jup_sto::ALLOW_INFINITY);
    chk1("nan", std::numeric_limits<real_t>::quiet_NaN(), false, jup_sto::ALLOW_NAN);
    chk2((real_t)3.14159265358979, (real_t)10.0, 1000);
    chk2((real_t)0.0, (real_t)1.0, 1000);
    chk2((real_t)0.0, (real_t)-1.0, 1000);
    chk2(std::numeric_limits<real_t>::max(), (real_t)0.0, 1000);
    chk2(std::numeric_limits<real_t>::lowest(), (real_t)0.0, 1000);
    chk2(std::numeric_limits<real_t>::min(), (real_t)0.0, 1000);
    chk2(std::numeric_limits<real_t>::min(), (real_t)1.0, 1000);
    chk3(1000000);
    chk4(1000000);

    if (warn_count) {
        jerr << "Warning: incorrect rounding (off-by-one) on " << warn_count << "/" << warn_total
             << " (" << jup_printf("%.5f%%", (real_t)warn_count / (real_t)warn_total) << ")\n";
    }
}

template <typename T>
void test_conversions_helper2(jup_str fmt, jup_str min, jup_str max) {
    Rng rng;
    
    auto chk1 = [fmt](jup_str str, T val, u16 code_target) {
        T i;
        auto code = jup_stox(str, &i);
        if (code != code_target) {
            jerr << "Error: code " << code << " while parsing " << str << " (expected " << code_target << ")\n";
            die();
        }
        if (code == 0 and i != val) {
            jerr << "Error: wrong parsing on " << str << ", expected " << jup_printf(fmt, val) << ", got "
                 << jup_printf(fmt, i) << "\n";
            die();
        }
    };
    auto chk2 = [&chk1](T val) {
        chk1(std::to_string(val), val, 0);
    };

    chk2(0);
    chk2(std::numeric_limits<T>::min());
    chk2(std::numeric_limits<T>::max());
    chk1(min, 0, 3);
    chk1(max, 0, 4);

    if ((u64)std::numeric_limits<T>::max() < 1000000) {
        for (T i = std::numeric_limits<T>::min();;) {
            chk2(i);
            if (__builtin_add_overflow(i, 1, &i)) break;
        }
    } else {
        for (int i = 0; i < 1000000; ++i) {
            chk2((T)rng.rand());
        }
    }
}

void test_conversions() {
    test_conversions_helper<double, u64, 300>("%.17e");
    test_conversions_helper<float,  u32, 30>("%.9e");
    test_conversions_helper2<u8>("%" PRIu8, "-1", "256");
    test_conversions_helper2<s8>("%" PRId8, "-129", "128");
    test_conversions_helper2<u16>("%" PRIu16, "-1", "65536");
    test_conversions_helper2<s16>("%" PRId16, "-32769", "32768");
    test_conversions_helper2<u32>("%" PRIu32, "-1", "4294967296");
    test_conversions_helper2<s32>("%" PRId32, "-2147483649", "2147483648");
    test_conversions_helper2<u64>("%" PRIu32, "-1", "18446744073709551616");
    test_conversions_helper2<s64>("%" PRId32, "-9223372036854775809", "9223372036854775808");
}

void test_histogram() {
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

} /* end of namespace jup */
