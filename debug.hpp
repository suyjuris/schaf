#pragma once

#include "array.hpp"
#include "buffer.hpp"
#include "graph.hpp"
#include "idmap.hpp"
#include "parse_alarm.hpp"
#include "utilities.hpp"

namespace jup {

void dbg_main();

#define __jup_get_macro(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9,\
_10, _11, _12, _13, _14, _15, _16, _17, _18, mac, ...) mac
#define __jup_fh1(mac, a) mac(a)
#define __jup_fh2(mac, a, ...) mac(a) __jup_fh1(mac, __VA_ARGS__)
#define __jup_fh3(mac, a, ...) mac(a) __jup_fh2(mac, __VA_ARGS__)
#define __jup_fh4(mac, a, ...) mac(a) __jup_fh3(mac, __VA_ARGS__)
#define __jup_fh5(mac, a, ...) mac(a) __jup_fh4(mac, __VA_ARGS__)
#define __jup_fh6(mac, a, ...) mac(a) __jup_fh5(mac, __VA_ARGS__)
#define __jup_fh7(mac, a, ...) mac(a) __jup_fh6(mac, __VA_ARGS__)
#define __jup_fh8(mac, a, ...) mac(a) __jup_fh7(mac, __VA_ARGS__)
#define __jup_fh9(mac, a, ...) mac(a) __jup_fh8(mac, __VA_ARGS__)
#define __jup_fh10(mac, a, ...) mac(a) __jup_fh9(mac, __VA_ARGS__)
#define __jup_fh11(mac, a, ...) mac(a) __jup_fh10(mac, __VA_ARGS__)
#define __jup_fh12(mac, a, ...) mac(a) __jup_fh11(mac, __VA_ARGS__)
#define __jup_fh13(mac, a, ...) mac(a) __jup_fh12(mac, __VA_ARGS__)
#define __jup_fh14(mac, a, ...) mac(a) __jup_fh13(mac, __VA_ARGS__)
#define __jup_fh15(mac, a, ...) mac(a) __jup_fh14(mac, __VA_ARGS__)
#define __jup_fh16(mac, a, ...) mac(a) __jup_fh15(mac, __VA_ARGS__)
#define __jup_fh17(mac, a, ...) mac(a) __jup_fh16(mac, __VA_ARGS__)
#define __jup_fh18(mac, a, ...) mac(a) __jup_fh17(mac, __VA_ARGS__)
#define __jup_fh19(mac, a, ...) mac(a) __jup_fh18(mac, __VA_ARGS__)
#define __forall(mac, ...) __jup_get_macro(__VA_ARGS__, __jup_fh19,\
__jup_fh18, __jup_fh17, __jup_fh16, __jup_fh15,__jup_fh14, __jup_fh13, __jup_fh12, __jup_fh11,\
__jup_fh10, __jup_fh9, __jup_fh8, __jup_fh7, __jup_fh6, __jup_fh5, __jup_fh4, __jup_fh3, __jup_fh2,\
 __jup_fh1, "fill") (mac, __VA_ARGS__)

#define __jup_sm1(x, y, ...) y
#define __jup_sm2(...) __jup_sm1(__VA_ARGS__, __jup_sm4,)
#define __jup_sm3(...) ~, __jup_sm5,
#define __jup_sm4(f1, f2, x) f1(x)
#define __jup_sm5(f1, f2, x) f2 x
#define __jup_select(f1, f2, x) __jup_sm2(__jup_sm3 x)(f1, f2, x)

// An output stream for debugging purposes
struct Debug_ostream {
    std::ostream& out;
    Buffer buf;
    Idmap* strings = nullptr;
    
    Debug_ostream(std::ostream& out): out{out} {}

    template <typename... Args>
    Debug_ostream& printf(char const* fmt, Args const&... args) {
        buf.reserve(256);
        while (true) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
            int count = std::snprintf(buf.data(), buf.capacity(), fmt, args...);
#pragma GCC diagnostic pop

            assert(count >= 0);
            if (count < buf.capacity()) break;
            buf.reserve(count);
        }
        out << buf.data();
        return *this;
    }

};

struct Debug_tabulator {
	u8 n;
	Debug_tabulator() {
		n = 0;
	}
};
extern Debug_tabulator tab;

inline Debug_ostream& operator< (Debug_ostream& out, Debug_tabulator const& tab) {
	for (u8 i = 0; i < tab.n; i++) {
		out.out << "    ";
	}
	return out;
}

struct Repr { Buffer_view data; };
inline Debug_ostream& operator< (Debug_ostream& out, Repr r) {
    out.out << '"';
	for (char c: r.data) {
        if (c == '\n') {
            out.out << "\\n";
        } else if (c == '\t') {
            out.out << "\\t";
        } else if (c == '\0') {
            out.out << "\\0";
        } else if (' ' <= c and c <= '~') {
            out.out << c;
        } else {
            out.printf("\\x%02hhx", (u8)c);
        }
	}
    out.out << "\" ";
	return out;
}

template <typename T>
struct Hex_t {
    T const& value;
};
template <typename T>
auto Hex(T const& val) { return Hex_t<T>{val}; }

template <typename T> struct Hex_fmt;
template <> struct Hex_fmt<u8>  { static constexpr char const* fmt = "0x%.2"  PRIx8;  };
template <> struct Hex_fmt<s8>  { static constexpr char const* fmt = "0x%.2"  PRIx8;  };
template <> struct Hex_fmt<u16> { static constexpr char const* fmt = "0x%.4"  PRIx16; };
template <> struct Hex_fmt<s16> { static constexpr char const* fmt = "0x%.4"  PRIx16; };
template <> struct Hex_fmt<u32> { static constexpr char const* fmt = "0x%.8"  PRIx32; };
template <> struct Hex_fmt<s32> { static constexpr char const* fmt = "0x%.8"  PRIx32; };
template <> struct Hex_fmt<u64> { static constexpr char const* fmt = "0x%.16" PRIx64; };
template <> struct Hex_fmt<s64> { static constexpr char const* fmt = "0x%.16" PRIx64; };
template <typename T> struct Hex_fmt<T*> { static constexpr char const* fmt = "%p"; };

template <typename T>
inline Debug_ostream& operator< (Debug_ostream& out, Hex_t<T> h) {
    out.printf(Hex_fmt<T>::fmt, h.value);
    out.out.put(' ');
    return out;
}

struct Id_string {
    Id_string(u32 id): id{id} {}
    u32 id;
};
inline Debug_ostream& operator< (Debug_ostream& out, Id_string i) {
    if (out.strings) {
        auto val = out.strings->get_value(i.id);
        out.out.put('"');
        out.out.write(val.data(), val.size());
        out.out.put('"');
        out.out.put(' ');
    } else {
        out < Hex(i.id);
    }
	return out;
}

template <typename T, T mask>
inline T apply_mask(T val) { return val & mask; }

inline void operator, (Debug_ostream& out, u8 n) {
	do {
		out.out << std::endl;
	} while (n --> 0);
}

template <typename T, typename T2, typename T3>
inline Debug_ostream& operator< (Debug_ostream& out, jup_str s) {
	out.out .write(s.data(), s.size()); return out < ' ';
}
template <typename T, typename T2, typename T3>
inline Debug_ostream& operator< (Debug_ostream& out, Flat_array<T, T2, T3> const& fa) {
	return out <= fa;
}
template <typename T>
inline Debug_ostream& operator< (Debug_ostream& out, Array<T> const& arr) {
	return out <= arr;
}
template <typename T>
inline Debug_ostream& operator< (Debug_ostream& out, Array_view<T> const& arr) {
	return out <= arr;
}
template <typename T>
inline Debug_ostream& operator< (Debug_ostream& out, Array_view_mut<T> const& arr) {
	return out <= arr;
}
template <typename T, size_t n>
Debug_ostream& operator< (Debug_ostream& out, T const (&arr)[n]) {
	return out <= arr;
}
template <typename T>
Debug_ostream& operator< (Debug_ostream& out, T const& obj) {
	out.out << obj << ' '; return out;
}
inline Debug_ostream& operator< (Debug_ostream& out, char const* s) {
	out.out << s << ' '; return out;
}
inline Debug_ostream& operator> (Debug_ostream& out, char const* s) {
    //for (auto i = s; *i; ++i) { out.out.put(*i != '\b' ? *i : '~'); }
	out.out << s;
    return out;
}
inline Debug_ostream& operator> (Debug_ostream& out, jup_str s) {
    out.out.write(s.data(), s.size());
    return out;
}
inline Debug_ostream& operator< (Debug_ostream& out, double d) {
	return out.printf("%.4e ", d);
}
inline Debug_ostream& operator< (Debug_ostream& out, float f) {
    return out < (double)f;
}
inline Debug_ostream& operator< (Debug_ostream& out, u8 n) {
    return out < (int)n;
}


template <typename T1, typename T2>
inline void print_nice(Debug_ostream& out, Flat_array_ref_base<T1, T2> const& ref, Buffer const& containing) {
    out > "(Flat_array_ref_base) {";
    out > "element_size = " < ref.element_size > "\b, ";
    out > "offset = " < ref.offset > "\b, ";
    //out > "first_byte = " < ref.first_byte(containing) > "\b, ";
    out > "last_byte = " < ref.last_byte(containing) > "\b";
    if (ref.name)
        out > ", name = " > ref.name;
    out > "} ";
}
template <typename T1, typename T2>
inline Debug_ostream& operator< (Debug_ostream& out, Diff_flat_arrays_base<T1, T2> const& diff) {
    out > "(Diff_flat_arrays) {";
    out > "container = " < make_hex(diff.container) > "\b, ";
    
    out > "refs = {\n";
    ++tab.n;
    for (auto const& ref: diff.refs()) {
        out < tab;
        print_nice(out, ref, *diff.container);
        out > "\b,\n";
    }
    --tab.n;
    out < tab > "} ";
    
    out > "diffs = {\n";
    ++tab.n;
    for (int i = diff.first(); i; diff.next(&i)) {
        u8 type = diff.diffs[i];
        u8 ref  = diff.diffs[i+1];
        if (type == Diff_flat_arrays::ADD) {
            Buffer_view data {diff.diffs.data() + i+2, diff.refs()[ref].element_size};
            out < tab > "type = ADD, ref = ";
            print_nice(out, diff.refs()[ref], *diff.container);
            out > "\b, data = " > nice_hex(data) > "\n";
        } else if (type == Diff_flat_arrays::REMOVE) {
            out < tab > "type = REMOVE, ref = ";
            print_nice(out, diff.refs()[ref], *diff.container);
            out > "\b, index = " < (int)diff.diffs[i+2] > "\n";
        } else {
            assert(false);
        }
    }
    --tab.n;
    out < tab > "} ";
	return out;
}

template <typename T>
inline Debug_ostream& operator> (Debug_ostream& out, T const& t) {
    return out < t;
}


extern Debug_ostream jdbg;

extern bool debug_flag;
#define JDBG_L (jdbg > __FILE__ ":" < __LINE__ > "\b: ")
#define JDBG_D if (debug_flag) JDBG_L

// type must have between 1 and 15 elements
#define __jup_display_var1(var) > " " > #var > " = " < obj.var > "\b,"
#define __jup_display_var2(var, fmt) > " " > #var > " = " < fmt(obj.var) > "\b,"
#define __jup_display_var(var) __jup_select(__jup_display_var1, __jup_display_var2, var)
#define __jup_display_val1(var) < obj.var > "\b, "
#define __jup_display_val2(var, fmt) < fmt(obj.var) > "\b, "
#define __jup_display_val(var) __jup_select(__jup_display_val1, __jup_display_val2, var)
#define __jup_display_obj(type, ...)                                          \
    out > "(" > #type > ") {" __forall(__jup_display_var, __VA_ARGS__) > "\b } "
#define __jup_display_vec(type, ...)                                          \
    out > "{" __forall(__jup_display_val, __VA_ARGS__) > "\b\b} "
#define __jup_print_gdb(type) \
    inline void print(type const& obj) __attribute__ ((used));  \
    inline void print(type const& obj) {                        \
        jup::jdbg < obj, 0;                                     \
    }
#define __jup_dbg(type, ...) \
    inline Debug_ostream& operator< (Debug_ostream& out, type const& obj) {    \
	    return __jup_display_obj(type, __VA_ARGS__);                           \
    }                                                                          \
    inline Debug_ostream& operator> (Debug_ostream& out, type const& obj) {    \
	    return __jup_display_vec(type, __VA_ARGS__);                           \
    }                                                                          \
    __jup_print_gdb(type)

#define __jup_hex(x) (x, Hex)
#define __jup_repr(x) (x, Repr)
#define __jup_id(x) (x, Id_string)
#define __jup_mask(x, m) (x, (apply_mask<decltype(m), m>))

__jup_dbg(Buffer, __jup_hex(m_data), m_size, __jup_mask(m_capacity, 0x7fffffff))
__jup_dbg(Idmap, m_size, data)
__jup_dbg(Git_object, type, __jup_hex(sha))
__jup_dbg(Git_tree_Entry, mode, __jup_hex(sha), __jup_id(name))
__jup_dbg(Git_commit, type, __jup_hex(sha), __jup_hex(tree), parents)
__jup_dbg(Git_tree, type, __jup_hex(sha), entries)
__jup_dbg(Alarm_stream, in_fd, in_data, in_data_off, in_data_znext, in_data_zstate, \
    out_data, strings, state)
__jup_dbg(Node, data_offset)
__jup_dbg(Edge, other, weight)

template <typename Range>
Debug_ostream& operator>= (Debug_ostream& out, Range const& r) {
	out > "{";
	if (std::begin(r) == std::end(r)) {
		return  out > "} ";
	}
    for (auto i = std::begin(r); i != std::end(r); ++i) {
        out > *i > "\b, ";
    }
    return out > "\b\b} ";
}

template <typename Range>
Debug_ostream& operator<= (Debug_ostream& out, Range const& r) {
	out > "{";
	if (std::begin(r) == std::end(r)) {
		return  out > "} ";
	}
	tab.n++;
    for (auto i = std::begin(r); i != std::end(r); ++i) {
        out > "\n" < tab < *i > "\b,";
    }
	tab.n--;
    return out > "\b \n" < tab > "} ";
}


} /* end of namespace jup */
