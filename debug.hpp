#pragma once

#include "array.hpp"
#include "buffer.hpp"
#include "idmap.hpp"
#include "parse_alarm.hpp"

namespace jup {

void dbg_main();

#define __get_macro(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9,\
_10, _11, _12, _13, _14, _15, _16, _17, _18, mac, ...) mac
#define __fh1(mac, a) mac(a)
#define __fh2(mac, a, ...) mac(a) __fh1(mac, __VA_ARGS__)
#define __fh3(mac, a, ...) mac(a) __fh2(mac, __VA_ARGS__)
#define __fh4(mac, a, ...) mac(a) __fh3(mac, __VA_ARGS__)
#define __fh5(mac, a, ...) mac(a) __fh4(mac, __VA_ARGS__)
#define __fh6(mac, a, ...) mac(a) __fh5(mac, __VA_ARGS__)
#define __fh7(mac, a, ...) mac(a) __fh6(mac, __VA_ARGS__)
#define __fh8(mac, a, ...) mac(a) __fh7(mac, __VA_ARGS__)
#define __fh9(mac, a, ...) mac(a) __fh8(mac, __VA_ARGS__)
#define __fh10(mac, a, ...) mac(a) __fh9(mac, __VA_ARGS__)
#define __fh11(mac, a, ...) mac(a) __fh10(mac, __VA_ARGS__)
#define __fh12(mac, a, ...) mac(a) __fh11(mac, __VA_ARGS__)
#define __fh13(mac, a, ...) mac(a) __fh12(mac, __VA_ARGS__)
#define __fh14(mac, a, ...) mac(a) __fh13(mac, __VA_ARGS__)
#define __fh15(mac, a, ...) mac(a) __fh14(mac, __VA_ARGS__)
#define __fh16(mac, a, ...) mac(a) __fh15(mac, __VA_ARGS__)
#define __fh17(mac, a, ...) mac(a) __fh16(mac, __VA_ARGS__)
#define __fh18(mac, a, ...) mac(a) __fh17(mac, __VA_ARGS__)
#define __fh19(mac, a, ...) mac(a) __fh18(mac, __VA_ARGS__)
#define __forall(mac, ...) __get_macro(__VA_ARGS__, __fh19,\
__fh18, __fh17, __fh16, __fh15,__fh14, __fh13, __fh12, __fh11,\
__fh10, __fh9, __fh8, __fh7, __fh6, __fh5, __fh4, __fh3, __fh2,\
 __fh1, "fill") (mac, __VA_ARGS__)

#define __sm1(x, y, ...) y
#define __sm2(...) __sm1(__VA_ARGS__, __sm4,)
#define __sm3(...) ~, __sm5,
#define __sm4(f1, f2, x) f1(x)
#define __sm5(f1, f2, x) f2 x
#define __select(f1, f2, x) __sm2(__sm3 x)(f1, f2, x)

// An output stream for debugging purposes
struct Debug_ostream {
    std::ostream& out;
    Buffer buf;
    Idmap* strings = nullptr;
    
    Debug_ostream(std::ostream& out): out{out} {}

    template <typename... Args>
    Debug_ostream& printf(c_str fmt, Args const&... args) {
        buf.reserve(256);
        while (true) {
            int count = std::snprintf(buf.data(), buf.capacity(), fmt, args...);
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

template <typename T> struct Hex { T const& value; };

template <typename T> auto make_hex(T const& obj) { return Hex<T> {obj}; }

template <typename T> struct Hex_fmt;
template <> struct Hex_fmt<u8>  { static constexpr c_str fmt = "0x%.2hhx"; };
template <> struct Hex_fmt<s8>  { static constexpr c_str fmt = "0x%.2hhx"; };
template <> struct Hex_fmt<u16> { static constexpr c_str fmt = "0x%.4hx"; };
template <> struct Hex_fmt<s16> { static constexpr c_str fmt = "0x%.4hx"; };
template <> struct Hex_fmt<u32> { static constexpr c_str fmt = "0x%.8x"; };
template <> struct Hex_fmt<s32> { static constexpr c_str fmt = "0x%.8x"; };
template <> struct Hex_fmt<u64> { static constexpr c_str fmt = "0x%.16I64x"; };
template <> struct Hex_fmt<s64> { static constexpr c_str fmt = "0x%.16I64x"; };
template <typename T> struct Hex_fmt<T*> { static constexpr c_str fmt = "%p"; };

template <typename T>
inline Debug_ostream& operator< (Debug_ostream& out, Hex<T> h) {
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
        out < make_hex(i.id);
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
template <typename T, size_t n>
Debug_ostream& operator< (Debug_ostream& out, T const (&arr)[n]) {
	return out <= arr;
}
template <typename T>
Debug_ostream& operator< (Debug_ostream& out, T const& obj) {
	out.out << obj << ' '; return out;
}
inline Debug_ostream& operator< (Debug_ostream& out, c_str s) {
	return out.printf(s);
}
inline Debug_ostream& operator< (Debug_ostream& out, double d) {
	return out.printf("%.2elf ", d);
}
inline Debug_ostream& operator< (Debug_ostream& out, float f) {
    return out < (double)f;
}
inline Debug_ostream& operator< (Debug_ostream& out, u8 n) {
    return out < (int)n;
}

extern Debug_ostream jdbg;

// type must have between 1 and 15 elements
#define display_var1(var) < " " < #var < " = " < obj.var < "\b,"
#define display_var2(var, fmt) < " " < #var < " = " < fmt(obj.var) < "\b,"
#define display_var(var) __select(display_var1, display_var2, var)
#define display_obj(type, ...)                                          \
    out < "(" < #type < ") {" __forall(display_var, __VA_ARGS__) < "\b } "
#define print_for_gdb(type) \
    inline void print(type const& obj) __attribute__ ((used));  \
    inline void print(type const& obj) {                        \
        jup::jdbg < obj, 0;                                     \
    }
#define op(type, ...) \
    inline Debug_ostream& operator< (Debug_ostream& out, type const& obj) { \
	    return display_obj(type, __VA_ARGS__);                              \
    }                                                                       \
    print_for_gdb(type)

#define hex(x) (x, make_hex)
#define repr(x) (x, Repr)
#define id(x) (x, Id_string)
#define mask(x, m) (x, (apply_mask<decltype(m), m>))

op(Buffer_view, hex(m_data), m_size)
op(Buffer, hex(m_data), m_size, mask(m_capacity, 0x7fffffff))
op(Idmap, m_size, data)
op(Git_object, type, hex(sha))
op(Git_tree_Entry, mode, hex(sha), id(name))
op(Git_commit, type, hex(sha), hex(tree), parents)
op(Git_tree, type, hex(sha), entries)
op(Alarm_stream, in_fd, in_data, in_data_off, in_data_znext, in_data_zstate, \
    out_data, strings, state)

#undef op
#undef display_obj
#undef display_var
#undef display_var1
#undef display_var2
#undef hex
#undef repr
#undef id
#undef mask

template <typename Range>
Debug_ostream& operator<= (Debug_ostream& out, Range const& r) {
	out < "{";
	if (std::begin(r) == std::end(r)) {
		return  out < "} ";
	}
	tab.n++;
    for (auto i = std::begin(r); i != std::end(r); ++i) {
        out < "\n" < tab < *i < "\b,";
    }
	tab.n--;
    return out < "\b \n" < tab < "} ";
}


} /* end of namespace jup */
