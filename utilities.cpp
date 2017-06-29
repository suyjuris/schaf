
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

jup_str nice_bytes(u64 bytes) {
    if (bytes < 1000) {
        return jup_printf("%d B", bytes);
    } else if ((bytes >> 10) < 1000) {
        return jup_printf("%.2f KiB", (float)bytes / 1024.f);
    } else if ((bytes >> 20) < 1000) {
        return jup_printf("%.2f MiB", (float)(bytes >> 10) / 1024.f);
    } else if ((bytes >> 30) < 1000) {
        return jup_printf("%.2f GiB", (float)(bytes >> 20) / 1024.f);
    } else {
        return jup_printf("%.2f TiB", (float)(bytes >> 30) / 1024.f);
    }
}

} /* end of namespace jup */
