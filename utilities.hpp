#pragma once

#include "arena.hpp"
#include "buffer.hpp"

namespace jup {

/**
 * Return a pointer to a region of memory at least size in size. It must not be
 * freed and is valid until the next call to tmp_alloc_reset.
 */
void* tmp_alloc(int size);

/**
 * Free memory allocated with tmp_alloc
 */
void tmp_alloc_reset();

/**
 * Return a buffer.
 */
Buffer& tmp_alloc_buffer();

/**
 * Return the memory used for tmp_alloc. This function does not allocate memory.
 */
Arena& tmp_alloc_arena();

template <typename T>
T&& string_unpacker(T&& obj) { return obj; }
inline char const* string_unpacker(jup_str obj) { return obj.c_str(); }

/**
 * Like sprintf, but uses tmp_alloc for memory. Additionally, it is allowed to pass null-terminated
 * jup_str, they will be converted into const char* before calling printf. Guaranteed to not modify
 * errno.
 */
template <typename... Args>
jup_str jup_printf(jup_str fmt, Args const&... args) {
    auto tmp_errno = errno;
    errno = 0;
    int size = std::snprintf(nullptr, 0, fmt.c_str(), string_unpacker(args)...);
    assert_errno(errno == 0);

    char* tmp = (char*)tmp_alloc(size + 1);
    
    errno = 0;
    std::snprintf(tmp, size + 1, fmt.c_str(), string_unpacker(args)...);
    assert_errno(errno == 0);

    errno = tmp_errno;
    
    return {tmp, size};
}

struct Std_free_functor {
    void operator() (void* p) { std::free(p); }
};

/**
 * std::unique_ptr support for memory allocated by std::malloc. Provides types Unique_ptr_free<T>
 * (just like a unique_ptr, but free'd using std::free), Unique_ptr_void (same as
 * Unique_prt_free<void>), and make_unique_void.
 */
template <typename T> using Unique_ptr_free = std::unique_ptr<T, Std_free_functor>;
using Unique_ptr_void = Unique_ptr_free<void>;
inline Unique_ptr_void make_unique_void(int size) {
    return Unique_ptr_void {std::malloc(size)};
}

/**
 * std::memcpy and std::memmove for ranges. Checks that the ranges have equal length.
 */
template <typename T1, typename T2>
void jup_memcpy(T1* into, T2 const& from) {
    int into_size = reinterpret_cast<char*>(into->end()) - (char*)into->begin();
    int from_size = (char const*)from.end() - (char const*)from.begin();
    assert(into_size == from_size);
    std::memcpy(into->begin(), from.begin(), into_size);
}
template <typename T1, typename T2>
void jup_memmove(T1* into, T2 const& from) {
    int into_size = reinterpret_cast<char*>(into->end()) - (char*)into->begin();
    int from_size = (char const*)from->end() - (char const*)from->begin();
    assert(into_size == from_size);
    std::memmove(into->begin(), from->begin(), into_size);
}
template <typename T1>
void jup_memset(T1* into, int val = 0) {
    int into_size = reinterpret_cast<char*>(into->end()) - (char*)into->begin();
    std::memset(into->begin(), val, into_size);
}
/**
 * code is one of ?errno, ?jup_errno or ?win (the latter is only supported on the windows platform). The result
 * will be a nicely formatted error message corresponding to the current value of this error code.
 */
jup_str get_error_msg(jup_str code);
jup_str _print_error_msg(jup_str msg);

/**
 * Prints msg onto the console as an error message and terminates the program.
 */
[[noreturn]] void die(jup_str msg);

/**
 * First, any leading error code specifiers are evaluated (see get_error_msg) and printed to the
 * console. Then, the remaining string is passed to jup_printf together with the rest of the
 * arguments, and then printed to the console as an error message. Then the program terminates.
 * Example:
 *     die("?errno while opening file %s", file_name)
 * would print
 *     Error: <errno error message>
 *     Error: while opening file <file_name>
 */
template <typename... Args>
[[noreturn]] void die(jup_str fmt, Args&&... args) {
    fmt = _print_error_msg(fmt);
    die(jup_printf(fmt, std::forward<Args>(args)...));
}

/**
 * Nicely formats a number of bytes, e.g. 9385211055596 becomes "8.54 TiB".
 */
jup_str nice_bytes(u64 bytes);

/** Nicely formats a number of seconds. Examples:
 *    value      | result
 *     90726e-12 |  "91ns"
 *     90726e-10 |  "9.07us"
 *     90726e-8  |  "0.91ms"
 *     90726e-6  |  "90.73ms"
 *     90726e-4  |  "9.07s"
 *     90726e-2  |  "00:15:07"
 *     90726     |  "1:01:12:06"
 */
jup_str nice_time(double seconds);

/**
 * Nicely formats bytes as either octal or hexadecimal bytes. If swap is true, the bytes are printed
 * in reverse order. (Which makes sense when printing numbers on a little endian machine.)
 */
jup_str nice_oct(Buffer_view data, bool swap = false);
jup_str nice_hex(Buffer_view data, bool swap = false);
template <typename T> jup_str nice_oct(T const& obj) {
    return nice_oct(Buffer_view::from_obj<T>(obj), true);
}
template <typename T> jup_str nice_hex(T const& obj) {
    return nice_hex(Buffer_view::from_obj<T>(obj), true);
}

/**
 * Prints the string str into the ostream out, while wrapping lines to the terminal's width,
 * preserving indentation.
 *
 * To be more precise, the string is split at newlines, disregarding everything after the last
 * newline. (In other words, str should end in a newline.) Each line is then split into an
 * indentation (the largest prefix consisting only of spaces), followed by a list of words
 * (separated by exactly one space). Then, the indendation followed by some words (such that they
 * fit the terminal's width) and a newline are printed, until no words remain. If a word is too big
 * to fit into one line, it will be split. If the terminal's width is smaller than 40, 40 is used
 * instead.
 *
 * Example:
 *   "The wonderful number Pi and its decimal representation\n   The number Pi is seen in many "
 *   "different applications. It can be approximated by 3.141592653589793238462643383279502884197, "
 *   "although that is not exact."
 *
 * In terminal of width 40: (The '|' are only for illustration):
 *   |The wonderful number Pi and its decimal |
 *   |representation                          |
 *   |   The number Pi is seen in many        |
 *   |   different applications. It can be    |
 *   |   approximated by                      |
 *   |   3.14159265358979323846264338327950288|
 *   |   4197, although that is not exact.    |
 */
void print_wrapped(std::ostream& out, jup_str str);

/**
 * An ostream similar to /dev/null.
 */
extern std::ostream& jnull;

/**
 * Strings for error messages.
 */
extern jup_str jup_err_messages[];
extern const int jup_err_messages_size;

/**
 * The last returned error code.
 */
extern int jup_errno;

/**
 * Flags for jup_stox. See jup_stox for documentation.
 */
namespace jup_sto {
    enum Parse_flags: u8 {
        // The values are subject to change.
        NONE = 0,
        ALLOW_INFINITY = 1,
        ALLOW_NAN = 2,
        _ONLY_INTEGER = 4,
        _MAX_LEFT = 8,
    };
}

/**
 * General number parsing routines. The string str is parsed (specifics depend on the value of
 * flags) and an error code is returned. If the code is 0, then the operation was successful, else
 * it is an index into the jup_err_messages array, where a user-friendly message can be found. Apart
 * from being zero/nonzero the error codes are not part of the public API of these functions and
 * subject to change.
 *
 * If the operation was successful, the parsed value is placed into the parameter into, else into
 * remains unmodified.
 *
 * The currently supported data types are 8/16/32/64-bit signed/unsigned integers and 32/64-bit
 * IEEE-754 floating point numbers.
 *
 * When parsing integers, currently no flags may be specified. The value will be represented
 * exactly, if possible, else an error will be raised. Only the value must be an integer, it may be
 * written in a form usually used for floating point numbers (e.g. 3.14159e5 is a valid).
 * 
 * When parsing floating point numbers, the flags jup_sto::ALLOW_INFINITY and jup_sto::ALLOW_NAN are
 * allowed. The enable parsing of the special values infinity and NaN, respectively. Parsing a
 * floating point number is not exact, of course. However, THIS FUNCTION IS NOT GUARATEED TO ROUND
 * CORRECTLY in general. This means that it is possible for the result to be off-by-one.
 * Empirically, round-trips (converting the number to a string with enough digits and then back)
 * work without fault, and parsing of random strings is wrong only ~0.00057% of the time for 64-bit
 * floats (it has not been observed to fail for 32-bit floats).
 *
 * The following formats are supported (all matching is case-insensitive):
 *   [+-]*(?=.)[0-9]*(\.[0-9]*)?(e[+-]*[0-9]+)?
 *     Base-10 number with optional fractional part and optional exponent.
 *   [+-]*0b(?=.)[01]*(.[01]*)?
 *     Base-2 number with optional fractional part
 *   [+-]*0(?=.)[0-7]*(.[0-7]*)?
 *     Base-8 number with optional fractional part
 *   [+-]*0x(?=.)[0-9a-f]*(.[0-9a-f]*)?
 *     Base-16 number with optional fractional part
 *   [+-]*(inf|infty|infinity)
 *     Infinity (for floating point values only, jup_sto::ALLOW_INFINITY flag must be set)
 *   [+-]*nan
 *     (quiet) NaN (for floating point values only, jup_sto::ALLOW_NAN flag must be set)
 *
 * Note that the (?=.) matches everything that is followed by at least one character, i.e. that is
 * not at the end of the string. To put it differently, the base specifier, either ("", "0b", "0" or
 * "0x") must not be followed by the end of the string.
 */
u16 jup_stox(jup_str str, u8*     into, u8 flags = 0);
u16 jup_stox(jup_str str, s8*     into, u8 flags = 0);
u16 jup_stox(jup_str str, u16*    into, u8 flags = 0);
u16 jup_stox(jup_str str, s16*    into, u8 flags = 0);
u16 jup_stox(jup_str str, u32*    into, u8 flags = 0);
u16 jup_stox(jup_str str, s32*    into, u8 flags = 0);
u16 jup_stox(jup_str str, u64*    into, u8 flags = 0);
u16 jup_stox(jup_str str, s64*    into, u8 flags = 0);
u16 jup_stox(jup_str str, float*  into, u8 flags = 0);
u16 jup_stox(jup_str str, double* into, u8 flags = 0);

struct Rng {
    constexpr static u64 init = 0xd1620b2a7a243d4bull;
    u64 rand_state = init;

    /**
     * Return a random u64
     */
    u64 rand();

    /**
     * Return the result of a Bernoulli-experiment with success rate perbyte/256
     */
    bool gen_bool(u8 perbyte = 128);

    /**
     * Generate a random value in [0, max)
     */
    u64 gen_uni(u64 max);
    
    /**
     * Return an exponentially distributed value, with parameter lambda = perbyte/256. Slow.
     */
    u8 gen_exp(u8 perbyte);

    /**
     * Generate a random float/double that represents a number.
     */
    float  gen_any_float();
    double gen_any_double();

    /**
     * Generic interface for the gen_any_* functions.
     */
    template <typename T> T gen_any();
    
    template <typename T>
    T const* choose_weighted(T const* ptr, int count) {
        if (count == 0) return nullptr;
        u64 sum = 0;
        for (auto const& i: Array_view<T> {ptr, count}) {
            sum += i.rating;
        }
        u64 x = gen_uni(sum);
        for (auto const& i: Array_view<T> {ptr, count}) {
            if (x < i.rating) return &i;
            x -= i.rating;
        }
        assert(false);
    }
};
template <> inline float  Rng::gen_any() { return gen_any_float();  }
template <> inline double Rng::gen_any() { return gen_any_double(); }

// Global instance of an Rng. Feel free to use as a replacement for std::rand.
extern Rng global_rng;

/**
 * Write the bytes of object obj and extra_bytes additional bytes into the file at path.
 */
template <typename T>
void save_bytes(jup_str path, T const& obj, int extra_bytes = 0) {
    assert(extra_bytes >= 0);
    std::ofstream o;
    o.open(path.c_str(), std::ios::out | std::ios::binary);
    assert_errno(o.good());
    o.write((char const*)&obj, sizeof(obj) + extra_bytes);
    assert_errno(o.good());
}

void load_bytes_object(jup_str path, char* into, int obj_size, int extra_bytes);
void load_bytes_buffer(jup_str path, Buffer* into, int maxsize = -1);

/**
 * Read the bytes from the file at path and treat it as an instance of obj. If extra_bytes is -1,
 * ignore any trailing bytes. Else, extra_bytes additional bytes will be read and written to the
 * memory at obj+1 and following. Callers must ensure that there is space!
 */
template <typename T>
void load_bytes(jup_str path, T* obj, int extra_bytes = 0) {
    load_bytes_object(path, (char*)obj, sizeof(T), extra_bytes);
}

u64 get_file_size(jup_str path);

/**
 * Convenience class for providing the user with status updates regarding an ongoing operation.
 */
struct Timer {
    // The minimum amount of time between status updates.
    constexpr static double update_delay = 5.f;

    /**
     * Initializes the Timer. progress_target should be the value at which the operation is complete
     * (default: 100), it is used when printing the percentage in progress.
     */
    Timer(u64 progress_target = 100);

    /**
     * Whether to print a status update now. If so, the timer is reset. Call this inside your outer
     * loop.
     */
    bool update();
    
    /**
     * Returns a percentage for the total progress (i.e. have/progress_target).
     */
    jup_str progress(u64 have);
    
    /**
     * Returns the rate of progress as bytes per second (nicely formatted, of course).
     */
    jup_str bytes(u64 have);

    /**
     * Returns the rate of progress in terms of counter per second
     */
    jup_str counter(u64 have);
    
    /**
     * Like bytes, but calculates the rate since starting the timer.
     */
    jup_str bytes_done(u64 total);

    /**
     * Returns the total amount of time for the operation. Call this after the operation is
     * finished.
     */
    jup_str total();

    u64 progress_target;
    u64 last_bytes;
    u64 last_counter;
    double start_time;
    double next_update;
    double cur_duration;
};

/**
 * Helper class for analyzing statistics on the fly. This is intended as a debugging utility to
 * determine the distribution of some numbers, without storing these numbers or making any
 * assumptions about their magnitude.
 *
 * This makes use of the p-squared algorithm, refer to "The P-Square Algorithm for Dynamic
 * Calculation of Percentiles and Histograms without Storing Observations" (Jain & Chlamtac, 1985)
 * for details.
 */
struct Histogram {
    /**
     * Initialized the Histogram. size is the number of categories to determine the positions
     * of. These categories are b-quantiles for equidistant values of b.
     */
    Histogram(int size = 100);

    /**
     * Call this for every point in your sample.
     */
    void add(float x);

    /**
     * Print a nicely-formatted histogram into jout. If the Histogram does not have enough data
     * points to be initialised, this operation failes with an informational message.
     */
    void print(int width = -1, int height = -1) const;
    
    void print_quant() const;
    void print_raw(jup_str fname) const;

    Unique_ptr_void _data;
    Array_view_mut<float> q_;
    Array_view_mut<int> n_;
    int n, b;
};

/**
 * Same as Histogram basically, except it doesn't use an approximation but actually stores all
 * values. Refer to Histogram's documentations.
 */
struct Histogram_exact {
    Histogram_exact(int size = 100);
    void add(float x);
    void print(int width = -1, int height = -1);
    void print_quant();
    void print_raw(jup_str fname);

    /**
     * Sorts the data and calculates the quantiles.
     */
    void calculate();
    
    Array<float> data;
    Array<float> q;
    int last_size = 0;
};

/**
 * Returns a formatted date and time in the form 2017-12-15_15-04-59. Uses tmp_alloc for storage.
 */
jup_str get_date_string(std::time_t timestamp = -1);

/**
 * Boilerplate code for iterating over f(x), with user-defined f and x in some range.
 */
template <typename Partial_viewer>
struct Partial_view_iterator {
    using difference_type = std::ptrdiff_t;
    using value_type = typename Partial_viewer::value_type;
    using pointer    = typename Partial_viewer::value_type*;
    using reference  = typename Partial_viewer::value_type&;
    using iterator_category = std::random_access_iterator_tag;
    
    using data_type = typename Partial_viewer::data_type;
    using _Self = Partial_view_iterator<Partial_viewer>;

    Partial_view_iterator(data_type* ptr): ptr{ptr} {}
    
    data_type* ptr;
    reference operator*  () const { return  Partial_viewer::view(*ptr); }
    pointer   operator-> () const { return &Partial_viewer::view(*ptr); }
    reference operator[] (difference_type n) const { return Partial_viewer::view(ptr[n]); }
    
    _Self& operator++ ()    { ++ptr; return *this; }
    _Self  operator++ (int) { _Self r = *this; ++ptr; return r; }
    _Self& operator-- ()    { --ptr; return *this; }
    _Self  operator-- (int) { _Self r = *this; --ptr; return r; }

    _Self& operator+= (difference_type n) { ptr += n; return *this; }
    _Self& operator-= (difference_type n) { ptr -= n; return *this; }
    _Self operator+ (difference_type n) const { _Self r = *this; return r += n; }
    _Self operator- (difference_type n) const { _Self r = *this; return r -= n; }
    
    bool operator== (_Self const o) const { return ptr == o.ptr; }
    bool operator!= (_Self const o) const { return ptr != o.ptr; }
    bool operator<  (_Self const o) const { return ptr <  o.ptr; }
    bool operator>  (_Self const o) const { return ptr >  o.ptr; }
    bool operator<= (_Self const o) const { return ptr <= o.ptr; }
    bool operator>= (_Self const o) const { return ptr >= o.ptr; }
};

template <typename Partial_viewer> Partial_view_iterator<Partial_viewer> operator+ (
    typename Partial_view_iterator<Partial_viewer>::difference_type n,
    Partial_view_iterator<Partial_viewer> p
) { return p + n; }
template <typename Partial_viewer> Partial_view_iterator<Partial_viewer> operator- (
    typename Partial_view_iterator<Partial_viewer>::difference_type n,
    Partial_view_iterator<Partial_viewer> p
) { return p - n; }

template <typename Partial_viewer>
struct Partial_view_range {
    using It = Partial_view_iterator<Partial_viewer>;

    template <typename T, int n>
    Partial_view_range(T (&range)[n]): _begin{std::begin(range)}, _end{std::end(range)} {}
    
    template <typename Range>
    Partial_view_range(Range range): _begin{std::begin(range)}, _end{std::end(range)} {}
    
    It _begin, _end;
    It begin() const { return _begin; }
    It end()   const { return _end;   }
};

} /* end of namespace jup */
