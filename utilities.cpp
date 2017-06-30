
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


jup_str nice_hex(Buffer_view data) {
    tmp_buffer.resize(data.size() * 2 + 1);

    for (int i = 0; i < data.size(); ++i) {
        char c1 = (u8)data[i] >> 4;
        char c2 = (u8)data[i] & 15;
        c1 = c1 < 10 ? c1 + '0' : c1 - 10 + 'a';
        c2 = c2 < 10 ? c2 + '0' : c2 - 10 + 'a';
        tmp_buffer[2*i]     = c1;
        tmp_buffer[2*i + 1] = c2;
    }
    tmp_buffer.back() = '\0';
    return tmp_buffer;
}

} /* end of namespace jup */
