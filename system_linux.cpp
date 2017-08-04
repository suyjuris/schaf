
#ifndef JUP_OS_LINUX
#error "You are trying to compile a *_linux file while not on Linux. Please check your build configuration."
#endif

#include "system.hpp"

#include "buffer.hpp"
#include "utilities.hpp"

namespace jup {

Buffer_view jup_exec(jup_str cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    assert_errno(pipe);

    tmp_alloc_buffer().reset();
    tmp_alloc_buffer().reserve_space(256);
    while (not std::feof(pipe)) {
        if (std::fgets(tmp_alloc_buffer().end(), tmp_alloc_buffer().capacity(), pipe)) {
            tmp_alloc_buffer().addsize(std::strlen(tmp_alloc_buffer().end()));
        }
    }
    pclose(pipe);
    return tmp_alloc_buffer();
}

void print_stacktrace() {
    jerr << "\nStack trace:\n";
    constexpr static int buffer_size = 128;
    void** trace = new void*[buffer_size];
    int got = backtrace(trace, buffer_size);
    if (got == buffer_size) {
        jerr << "Warning: Insufficient buffer size while printing stacktrace\n";
    }

    char** messages = backtrace_symbols(trace, got);
    for (int i = 0; i < got; ++i) {
        int n = std::strlen(messages[i]);
        char* exe_name = (char*)alloca(n);
        char* fun_name = (char*)alloca(n);
        u64 fun_offset;
        std::sscanf(messages[i], "%[^(](%[^+]+%lx", exe_name, fun_name, &fun_offset);

        if (std::strcmp(fun_name, "main") == 0) got = i;
        
        auto res = jup_exec(jup_printf("nm %s 2>/dev/null | grep -F ' %s'", exe_name, fun_name));
        if (not res.size()) continue;
        
        u64 sym_offset;
        std::sscanf(res.c_str(), "%lx ", &sym_offset);
        
        auto str = jup_exec(jup_printf("addr2line -C -f -e %s 0x%lx", exe_name, sym_offset + fun_offset));
        if (not str.size()) continue;

        char* dem_name = (char*)alloca(str.size());
        char* fil_name = (char*)alloca(str.size());
        int line = -1;
        std::sscanf(str.c_str(), "%[^\n]\n%[^:]:%d\n", dem_name, fil_name, &line);

        for (int j = std::strlen(dem_name) - 1; j >= 0; --j) {
            if (dem_name[j] == '(') dem_name[j] = 0;
        }

        if (std::strcmp(dem_name, "jup::print_stacktrace") == 0) continue;
        if (std::strcmp(dem_name, "jup::die") == 0) continue;
        if (std::strcmp(dem_name, "jup::_assert_errno_fail") == 0) continue;
        
        if (line == -1) {
            jerr << "  (filename not available): " << dem_name << '\n';
        } else {
            char* pwd = get_current_dir_name();
            if (std::strncmp(pwd, fil_name, std::strlen(pwd)) == 0 and std::strlen(pwd) > 1) {
                fil_name += std::strlen(pwd) - 1;
                fil_name[0] = '.';
            }
            free(pwd);
            jerr << "  " << fil_name << ":" << line << ": " << dem_name << '\n';
        }
    }
    free(messages);
}

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

static bool is_currently_dying = false;

void die() {
    if (not is_currently_dying) {
        is_currently_dying = true;
        print_stacktrace();
    }
    
    std::abort();
}

int get_terminal_width() {
    struct winsize size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);

    return size.ws_col;
}

} /* end of namespace jup */
