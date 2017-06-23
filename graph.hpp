#pragma once

#include "flat_data.hpp"
#include "idmap.hpp"

namespace jup {

struct Graph {
    Flat_array<u32, u32, u32> data;
};

void graph_exec_jobfile(jup_str file);

} /* end of namespace jup */
