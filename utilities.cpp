
#include "utilities.hpp"

namespace jup {

static Buffer tmp_buffer;

// see header
void* tmp_alloc(int size) {
    tmp_buffer.reserve(size);
    return tmp_buffer.data();
}

// see header
Buffer& tmp_alloc_buffer() {
    return tmp_buffer;
}


} /* end of namespace jup */
