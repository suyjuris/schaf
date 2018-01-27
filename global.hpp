
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
#include <cinttypes>
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
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef __has_include
#if __has_include(<filesystem>)
#include <filesystem>
namespace std__filesystem = std::filesystem;
#define __JUP_FOUND_FILESYSTEM
#endif
#endif
#ifndef __JUP_FOUND_FILESYSTEM
#include <experimental/filesystem>
namespace std__filesystem = std::experimental::filesystem;
#endif

// zlib
#include <zlib.h>

// system libraries
#ifdef JUP_OS_WINDOWS
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <execinfo.h>
#endif

#ifdef NDEBUG

#define __jup_assert(expr) (void)__builtin_expect(not (expr), 0)
#define assert_errno(expr) assert(expr)

#ifdef JUP_OS_WINDOWS
#define assert_win(expr) assert(expr)
#endif

#else

#define __jup_assert(expr) ((expr) ? (void)0 : ::jup::_assert_fail(#expr, __FILE__, __LINE__))
#define assert_errno(expr) ((expr) ? (void)0 : ::jup::_assert_errno_fail(#expr, __FILE__, __LINE__))

#ifdef JUP_OS_WINDOWS
#define assert_win(expr) ((expr) ? (void)0 : ::jup::_assert_win_fail(#expr, __FILE__, __LINE__))
#endif

#endif

#undef assert
#define assert __jup_assert

#define __JUP_UNIQUE_NAME1(x, y) x##y
#define __JUP_UNIQUE_NAME2(x, y) __JUP_UNIQUE_NAME1(x, y)
#define JUP_UNIQUE_NAME(x) __JUP_UNIQUE_NAME2(x, __COUNTER__)

#define __JUP_STRINGIFY1(x) #x
#define JUP_STRINGIFY(x) __JUP_STRINGIFY1(x)

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
[[noreturn]] void die();

// Registers a signal handler to print things nicely
void init_signals();

// Registers the SIGINT handler, which sets the flag on receiving the first SIGINT. (The second one
// terminates the program.) Use this if you want to handle SIGINTs gracefully.
extern bool global_interrupt_flag;
void init_signal_sigint();

// Use these facilities for general output. They may redirect into a logfile later on.
extern std::ostream& jout;
extern std::ostream& jerr;
using std::endl;

} /* end of namespace jup */
