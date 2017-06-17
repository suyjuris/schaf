// Needed for CancelSynchronousIo
#undef _WIN32_WINNT
#define _WIN32_WINNT _WIN32_WINNT_VISTA

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

// general headers
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>   
#include <memory>
#include <ostream>
#include <string>
#include <vector>

// zlib
#include <zlib.h>

// win32 libraries
#include <windows.h>

#define assert(expr) ((expr) ? (void)0 : jup::_assert_fail(#expr, __FILE__, __LINE__))
#define assert_errno(expr) ((expr) ? (void)0 : jup::_assert_errno_fail(#expr, __FILE__, __LINE__))
// win32 specific
#define assert_win(expr) ((expr) ? (void)0 : jup::_assert_win_fail(#expr, __FILE__, __LINE__))

namespace jup {

// Standard integer types
using s64 = std::int64_t;
using u64 = std::uint64_t;
using s32 = std::int32_t;
using u32 = std::uint32_t;
using s16 = std::int16_t;
using u16 = std::uint16_t;
using s8 = std::int8_t;
using u8 = std::uint8_t;

// Zero terminated, read-only string
using c_str = char const*;

// Custom assertions, prints stack trace
void _assert_fail(c_str expr_str, c_str file, int line);
void _assert_errno_fail(c_str expr_str, c_str file, int line);
// win32 specific
void _assert_win_fail(c_str expr_str, c_str file, int line);

// Prints the error nicely into the console
void err_msg(c_str msg, int code = 0);

// Narrow a value, asserting that the conversion is valid.
template <typename T, typename R>
inline void narrow(T& into, R from) {
	into = static_cast<T>(from);
	assert(static_cast<R>(into) == from and (into > 0) == (from > 0));
}

// Closes the program violently
void die();
void die(c_str msg, int code = 0);

// Use these facilities for general output. They may redirect into a logfile later on.
extern std::ostream& jout;
extern std::ostream& jerr;
using std::endl;

} /* end of namespace jup */
