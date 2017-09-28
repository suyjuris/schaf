#pragma once

namespace jup {

struct Network_state;

Network_state* network_init();
void network_free(Network_state* state);
void network_main();

extern constexpr int batch_count = 17; // batches per training data
extern constexpr int batch_size  = 13; // instances per batch
extern constexpr int batch_nodes = 16; // nodes per instance

extern constexpr int gen_graph_nodes = 32; // Nodes needed per instance

struct Batch_data {
    Array_inline<float, batch_size * batch_nodes * batch_nodes> edge_weights;
    Array_inline<float, batch_size> results;
};

struct Training_data {
    Array_inline<Batch_data, batch_count> batches;
};

void network_gendata(jup_str graph_file, Training_data* data);
void network_prepare(jup_str graph_file, jup_str data_file);

} /* end of namespace jup */
