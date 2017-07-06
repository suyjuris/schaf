
#ifndef JUP_OS_LINUX
#error "You are trying to compile a *_linux file while not on Linux. Please check your build configuration."
#endif

#include "system.hpp"

#include <sys/ioctl.h>

namespace jup {

void die() {    
    std::abort();
}

int get_terminal_width() {
    struct winsize size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);

    return size.ws_col;
}

} /* end of namespace jup */
