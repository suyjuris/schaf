
#include "utilities.hpp"

namespace jup {

std::ostream& jout = std::cout;
std::ostream& jerr = std::cerr;

void _assert_fail(char const* expr_str, char const* file, int line) {
    jerr << "\nError: Assertion failed. File: " << file << ", Line " << line
         << "\n\nExpression: " << expr_str << "\n";
    die();
}

void _assert_errno_fail(char const* expr_str, char const* file, int line) {
    jerr << "\nError: Assertion failed. File: " << file << ", Line " << line
         << "\n\nExpression: " << expr_str << "\n";
    die("?errno");
}

extern "C" void handle_signal(int sig) {
    char const* msg;
    switch (sig) {
    case SIGINT:  msg = "Caught interrupt\n";      break;
    case SIGTERM: msg = "Caught a SIGTERM\n";      break;
    case SIGILL:  msg = "Caught a SIGILL\n";       break;
    case SIGSEGV: msg = "Segmentation fault\n";    break;
    case SIGFPE:  msg = "Arithmetic exception\n";  break;
    default:      msg = "Unknwon signal caught\n"; break;
    }
    die(msg, sig);
}

void init_signals() {
    assert(std::signal(SIGINT,  &handle_signal) != SIG_ERR);
    assert(std::signal(SIGTERM, &handle_signal) != SIG_ERR);
    assert(std::signal(SIGILL,  &handle_signal) != SIG_ERR);
    assert(std::signal(SIGSEGV, &handle_signal) != SIG_ERR);
    assert(std::signal(SIGFPE,  &handle_signal) != SIG_ERR);
}

} /* end of namespace jup */

