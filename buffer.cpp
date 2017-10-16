
#include "libs/xxhash.hpp"

#include "buffer.hpp"

namespace jup {

u64 Buffer_view::get_hash() const {
    static const u64 base = XXH64("", 0, 0);
    return XXH64(begin(), size(), 0) ^ base;
}

} /* end of namespace jup */
