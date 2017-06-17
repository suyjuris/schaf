#pragma once

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

// An output stream for debugging purposes
struct Debug_ostream {
    std::ostream& out;
    Buffer buf;
    
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

struct Repr {
    Buffer_view data;
};

inline Debug_ostream& operator< (Debug_ostream& out, Debug_tabulator const& tab) {
	for (u8 i = 0; i < tab.n; i++) {
		out.out << "    ";
	}
	return out;
}
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
            out.printf("\\x%02x", c);
        }
	}
    out.out << "\" ";
	return out;
}

inline void operator, (Debug_ostream& out, u8 n) {
	do {
		out.out << std::endl;
	} while (n --> 0);
}

template <typename T, typename T2, typename T3>
inline Debug_ostream& operator< (Debug_ostream& out, Flat_array<T, T2, T3> const& fa) {
	return out <= fa;
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
#define display_var(var) < " " < #var < " = " < obj.var< "\b,"
#define display_obj(type, ...)                                          \
    out < "(" < #type < ") {" __forall(display_var, __VA_ARGS__) < "\b } "
#define display_obJ(type, id, ...)                                       \
    out < "(" < #type < ") {" < #id < " = " < (obj.id ?                  \
    get_string_from_id(obj.id).c_str() : "") \
     < "," __forall(display_var, __VA_ARGS__) < "\b } "
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

#define oP(type, id, ...)                                                   \
    inline Debug_ostream& operator< (Debug_ostream& out, type const& obj) { \
        return display_obJ(type, id, __VA_ARGS__);                          \
    }                                                                       \
    print_for_gdb(type)

op(Buffer_view, m_data, m_size)
op(Buffer, m_data, m_size, m_capacity)
op(Idmap, m_size, data)
op(Git_object, type, sha)
op(Git_tree_Entry, mode, name, sha)
op(Git_commit, tree, parents)
op(Git_tree, entries)
op(Alarm_stream, in_fd, in_data, in_data_off, in_data_znext, in_data_zstate, \
    out_data, strings, state)

#undef op
#undef oP
#undef display_obj
#undef display_var_ex
#undef display_var

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
