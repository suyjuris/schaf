#pragma once

#include "arena.hpp"
#include "network.hpp"

namespace jup {

#define JUP_DEFAULT_ITER_SAVE 5000
#define JUP_DEFAULT_ITER_EVENT 100
#define JUP_DEFAULT_LOGDIR "tf_data"

struct Schaf_options {
    int graph_min_edges = 0;
    int graph_max_edges = std::numeric_limits<int>::max();
    Hyperparam hyp;
    jup_str param_in;
    int iter_max = std::numeric_limits<int>::max();
    int iter_save = JUP_DEFAULT_ITER_SAVE;
    int iter_event = JUP_DEFAULT_ITER_EVENT;
    jup_str logdir = JUP_DEFAULT_LOGDIR;

    Arena string_storage;
};

extern Schaf_options global_options;

void options_execute(Schaf_options* options, Array_view<jup_str> args);

} /* end of namespace jup */
