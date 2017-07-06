#pragma once

#include "allocator.hpp"

namespace jup {

struct Schaf_options {
    enum Mode: u8 {
        INVALID = 0, WRITE_GRAPH, PRINT_STATS
    };

    u8 mode = INVALID;
    int graph_min_edges = 0;
    int graph_max_edges = std::numeric_limits<int>::max();

    Arena_allocator string_storage;
};

extern Schaf_options global_options;

void options_execute(Schaf_options* options, Array_view<jup_str> args);

} /* end of namespace jup */
