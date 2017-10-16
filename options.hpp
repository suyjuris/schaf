#pragma once

#include "arena.hpp"
#include "network.hpp"

namespace jup {

#define JUP_DEFAULT_ITER_SAVE 10

struct Schaf_options {
    enum Mode: u8 {
        INVALID = 0, WRITE_GRAPH, PRINT_STATS
    };

    // See the help text for documentation
    u8 mode = INVALID;
    int graph_min_edges = 0;
    int graph_max_edges = std::numeric_limits<int>::max();
    Hyperparam hyp;
    jup_str param_in;
    jup_str param_out;
    int iter_max = std::numeric_limits<int>::max();
    int iter_save = JUP_DEFAULT_ITER_SAVE;

    Arena string_storage;
};

extern Schaf_options global_options;

void options_execute(Schaf_options* options, Array_view<jup_str> args);

} /* end of namespace jup */
