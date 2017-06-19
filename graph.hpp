#pragma once

#include "flat_data.hpp"
#include "idmap.hpp"

namespace jup {

struct Graph {
    Flat_array<u32, u32, u32> data;
};

} /* end of namespace jup */
