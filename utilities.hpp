#pragma once

#include "buffer.hpp"

namespace jup {

/**
 * Return a pointer to a region of memory at least size in size. It must not be
 * freed and is valid until the next call to this function. DON'T USE IT TWICE
 * IN THE SAME OUTPUT LINE!
 *     jout << jup_printf("%d", i) << " and then " << jup_printf("%d", j);
 * This will only print one of these strings correctly, depending on the order
 * of evaluation of operands. Instead, do:
 *     jout << jup_printf("%d and then %d", i, j);
 */
void* tmp_alloc(int size);

/**
 * Return the memory used for the last call to tmp_alloc. This function does not
 * allocate memory.
 */
Buffer& tmp_alloc_buffer();

template <typename T>
T&& string_unpacker(T&& obj) { return obj; }
inline char const* string_unpacker(jup_str obj) { return obj.c_str(); }

/**
 * Like sprintf, but uses tmp_alloc for memory.
 */
template <typename... Args>
jup_str jup_printf(jup_str fmt, Args const&... args) {
    auto& buf = tmp_alloc_buffer();

    errno = 0;
    int size = std::snprintf(buf.data(), buf.capacity(), fmt.c_str(), string_unpacker(args)...);
    assert_errno(errno == 0);
    
    if (size >= buf.capacity()) {
        buf.reserve(size + 1);

        errno = 0;
        int size = std::snprintf(buf.data(), buf.capacity(), fmt.c_str(), string_unpacker(args)...);
        assert_errno(errno == 0);
        assert(size < buf.capacity());
    }
    return {buf.data(), size};
}

jup_str nice_bytes(u64 bytes);

jup_str nice_oct(Buffer_view data, bool swap = false);
jup_str nice_hex(Buffer_view data);

template <typename T>
jup_str nice_oct(T const& obj) {
    return nice_oct(Buffer_view::from_obj<T>(obj), true);
}
template <typename T>
jup_str nice_hex(T const& obj) {
    return nice_hex(Buffer_view::from_obj<T>(obj));
}

void print_wrapped(std::ostream& out, jup_str str);

extern std::ostream& jnull;

extern jup_str jup_stoi_messages[];
u8 jup_stoi(jup_str str, int* val);


} /* end of namespace jup */
