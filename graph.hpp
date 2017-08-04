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
    Flat_array<char, u32, u32> name;
    Flat_array<Node, u32, u32> nodes;
    Flat_array<Edge, u32, u32> edge_data;

    int num_nodes() const { return nodes.size() - 1; }
    int num_edges() const { return edge_data.size() / 2; }

    Array_view<Edge> adjacent(u32 node_id) const {
        Edge const* begin = edge_data.begin() + nodes[node_id  ].data_offset;
        Edge const* end   = edge_data.begin() + nodes[node_id+1].data_offset;
        return Array_view<Edge> {begin, narrow<int>(end - begin)};
    }

    constexpr static int total_space(int name_size, int num_nodes, int num_edges) {
        return sizeof(Graph)
            + decltype(name)::extra_space(name_size + 1)
            + decltype(nodes)::extra_space(num_nodes + 1)
            + decltype(edge_data)::extra_space(num_edges * 2);
    }
};

void graph_exec_jobfile(jup_str file, jup_str output);
void graph_print_stats(jup_str input, jup_str output);

struct Graph_reader_state {    
    Buffer data;
    std::ifstream input;
    Graph* graph = nullptr;
    double duration = 0;
    u32 graph_size = 0;
    u32 lz4_size = 0;
    u64 hash_val = 0;
};

void graph_reader_init(Graph_reader_state* state, jup_str file);
bool graph_reader_next(Graph_reader_state* state);
void graph_reader_close(Graph_reader_state* state);

} /* end of namespace jup */
