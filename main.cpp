
#include "graph.hpp"
#include "network.hpp"
#include "options.hpp"
#include "parse_alarm.hpp"
#include "system.hpp"

using namespace jup;

int main(int argc, char const* const* argv) {
    init_signals();
    init_elapsed_time();
    
    network_main();
    
/*
    Array<jup_str> argv_arr;
    for (int i = 1; i < argc; ++i) {
        argv_arr.push_back(argv[i]);
    }

    options_execute(&global_options, argv_arr);
*/
	return 0;
}
