
// Note: The Makefile should define JUP_OS as either Linux or Windows, depending on the OS. Additionally, either JUP_OS_WINDOWS or JUP_OS_LINUX

#ifdef JUP_OS_WINDOWS
// Needed for CancelSynchronousIo (so, actually unneeded)
//#undef _WIN32_WINNT
//#define _WIN32_WINNT _WIN32_WINNT_VISTA
#endif

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

// general headers
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
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
#ifdef JUP_OS_WINDOWS
#include <windows.h>
#endif

#ifdef NDEBUG

#define assert(expr) (void)__builtin_expect(not (expr), 0)
#define assert_errno(expr) assert(expr)

#ifdef JUP_OS_WINDOWS
#define assert_win(expr) assert(expr)
#endif

#else

#define assert(expr) ((expr) ? (void)0 : ::jup::_assert_fail(#expr, __FILE__, __LINE__))
#define assert_errno(expr) ((expr) ? (void)0 : ::jup::_assert_errno_fail(#expr, __FILE__, __LINE__))

#ifdef JUP_OS_WINDOWS
#define assert_win(expr) ((expr) ? (void)0 : ::jup::_assert_win_fail(#expr, __FILE__, __LINE__))
#endif

#endif

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

// Custom assertions, prints stack trace
[[noreturn]] void _assert_fail(char const* expr_str, char const* file, int line);
[[noreturn]] void _assert_errno_fail(char const* expr_str, char const* file, int line);

#ifdef JUP_OS_WINDOWS
[[noreturn]] void _assert_win_fail(char const* expr_str, char const* file, int line);
#endif

// Prints the error nicely into the console
void err_msg(char const* msg, int code = 0);

// Narrow a value, asserting that the conversion is valid.
template <typename T, typename R>
inline void narrow(T& into, R from) {
	into = static_cast<T>(from);
	assert(static_cast<R>(into) == from and (into > 0) == (from > 0));
}
template <typename T, typename R>
inline T narrow(R from) {
	T result = static_cast<T>(from);
	assert(static_cast<R>(result) == from and (result > 0) == (from > 0));
    return result;
}

// Closes the program violently
[[noreturn]] void die(); // implemented in system_win32.cpp
[[noreturn]] void die(char const* msg, int code = 0);

// Registers a signal handler to print things nicely
void init_signals();

// Use these facilities for general output. They may redirect into a logfile later on.
extern std::ostream& jout;
extern std::ostream& jerr;
using std::endl;

} /* end of namespace jup */
