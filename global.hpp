// Needed for CancelSynchronousIo
#undef _WIN32_WINNT
#define _WIN32_WINNT _WIN32_WINNT_VISTA

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef __USE_W32_SOCKETS
#define __USE_W32_SOCKETS
#endif


// general headers
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

// win32 libraries
#include <windows.h>

#include "stack_walker.hpp"

#define assert(expr) ((expr) ? (void)0 : jup::_assert_fail(#expr, __FILE__, __LINE__))

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

// Custom assertion, prints stack trace
void _assert_fail(c_str expr_str, c_str file, int line);

// Narrow a value, asserting that the conversion is valid.
template <typename T, typename R>
inline void narrow(T& into, R from) {
	into = static_cast<T>(from);
	assert(static_cast<R>(into) == from and (into > 0) == (from > 0));
}

// Closes the program
void die();

// Use these facilities for general output. They may redirect into a logfile later on.
extern std::ostream& jout;
extern std::ostream& jerr;
using std::endl;

} /* end of namespace jup */
