#pragma once

#include "flat_data.hpp"
#include "utilities.hpp"

namespace jup {
#define JUP_DEFAULT_BATCH_COUNT      256
#define JUP_DEFAULT_BATCH_SIZE        64
#define JUP_DEFAULT_BATCH_NODES       16
#define JUP_DEFAULT_GEN_GRAPH_NODES   32
#define JUP_DEFAULT_LEARNING_RATE   0.1f

struct Hyperparam {
    int batch_count = JUP_DEFAULT_BATCH_COUNT; // batches per training data
    int batch_size  = JUP_DEFAULT_BATCH_SIZE;  // instances per batch
    int batch_nodes = JUP_DEFAULT_BATCH_NODES; // nodes per instance
    
    int gen_graph_nodes = JUP_DEFAULT_GEN_GRAPH_NODES; // nodes needed per instance

    float learning_rate = JUP_DEFAULT_LEARNING_RATE; // you should know what that means...

    int num_instances() const {
        return batch_count * batch_size;
    }
    int floats_edge_weights() const {
        return batch_size * batch_nodes * batch_nodes;
    }
    int floats_results() const {
        return batch_size;
    }
    int floats_batch() const {
        return floats_edge_weights() + floats_results();
    }
    int floats_total() const {
        return floats_batch() * batch_count;
    }
    int bytes_batch() const {
        return floats_batch() * sizeof(float);
    }
    int bytes_instance() const {
        return floats_batch() / batch_size * sizeof(float);
    }
    int bytes_total() const {
        return floats_total() * sizeof(float);
    }
    bool valid() const {
        return batch_count > 0 and batch_size > 0 and batch_nodes > 0;
    }
};

inline std::ostream& operator<< (std::ostream& out, Hyperparam hyp) {
    out << "batch_count: " << hyp.batch_count << ", batch_size: " << hyp.batch_size
        << ", batch_nodes: " << hyp.batch_nodes << ", gen: " << hyp.gen_graph_nodes;
    return out;
}

struct Batch_data {
    Array_view_mut<float> edge_weights;
    Array_view_mut<float> results;
};

struct Training_data {
    Hyperparam hyp;
    Flat_array64_const<float> batch_data;

    static Unique_ptr_free<Training_data> make_unique(Hyperparam hyp);

    Batch_data batch(int index);
    Batch_data instance(int index);
    
    int byte_size() const {
        return sizeof(Training_data) + hyp.bytes_total();
    }
};

struct Network_state;

Network_state* network_init();
void network_free(Network_state* state);
void network_main();

void network_prepare_data(jup_str graph_file, jup_str data_file, Hyperparam hyp);
//void network_shuffle_data(jup_str from_file, jup_str into_file, Hyperparam into_hyp);

} /* end of namespace jup */
