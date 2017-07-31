
#ifndef JUP_OS_LINUX
#error "You are trying to compile a *_linux file while not on Linux. Please check your build configuration."
#endif

#include "system.hpp"

namespace jup {

static double elapsed_time_offset;

double elapsed_time() {
    timespec t;
    assert_errno( clock_gettime(CLOCK_MONOTONIC, &t) == 0 );
    return (t.tv_sec + 1e-9 * t.tv_nsec) - elapsed_time_offset;
}

void init_elapsed_time(double val) {
    elapsed_time_offset = 0;
    elapsed_time_offset = elapsed_time() - val;
}

void die() {    
    std::abort();
}

int get_terminal_width() {
    struct winsize size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);

    return size.ws_col;
}

} /* end of namespace jup */
