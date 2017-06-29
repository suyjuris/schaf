#pragma once

#include "array.hpp"
#include "flat_data.hpp"

namespace jup {

struct Node {
    u32 data_offset;
};

struct Edge {
    u32 other;
    u32 weight;
};

struct Graph {
    Flat_array<Node, u32, u32> nodes;
    Flat_array<Edge, u32, u32> edge_data;

    int num_nodes() const { return nodes.size() - 1; }
    int num_edges() const { return edge_data.size() / 2; }

    Array_view<Edge> adjacent(u32 node_id) {
        Edge* begin = &edge_data[nodes[node_id  ].data_offset];
        Edge* end   = &edge_data[nodes[node_id+1].data_offset];
        return Array_view<Edge> {begin, narrow<int>(end - begin)};
    }

    constexpr static int total_space(int num_nodes, int num_edges) {
        return sizeof(Graph) + decltype(nodes)::extra_space(num_nodes + 1)
            + decltype(edge_data)::extra_space(num_edges * 2);
    }
};

void graph_exec_jobfile(jup_str file, jup_str output);

} /* end of namespace jup */
