#pragma once
 
#include "flat_data.hpp"
#include "utilities.hpp"

namespace jup {

#define JUP_DEFAULT_BATCH_COUNT      256
#define JUP_DEFAULT_BATCH_SIZE        64
#define JUP_DEFAULT_BATCH_NODES       16
#define JUP_DEFAULT_GEN_GRAPH_NODES   32
#define JUP_DEFAULT_LEARNING_RATE    0.1
#define JUP_DEFAULT_TEST_FRAC        0.1

struct Hyperparam {
    int batch_count = JUP_DEFAULT_BATCH_COUNT; // batches per training data
    int batch_size  = JUP_DEFAULT_BATCH_SIZE;  // instances per batch
    int batch_nodes = JUP_DEFAULT_BATCH_NODES; // nodes per instance
    
    int gen_graph_nodes = JUP_DEFAULT_GEN_GRAPH_NODES; // nodes needed per instance

    float learning_rate = (float)JUP_DEFAULT_LEARNING_RATE; // you should know what that means...
    int learning_rate_decay = 0; // number of epochs after which the learning rate is halved.

    float test_frac = (float)JUP_DEFAULT_TEST_FRAC; // amount of training data to use as test data

    int a1_size = 64; // size of the output of the first layer
    int a2_size =  1; // size of the output of the second layer (has to be 1, if it is the last layer)

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
    int batch_edges() const {
        return batch_nodes * batch_nodes;
    }
} __attribute__((__packed__));

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
    Hyperparam hyp; // This must be in front!
    Flat_array64_const<float> batch_data;

    static Unique_ptr_free<Training_data> make_unique(Hyperparam hyp);

    Batch_data batch(int index);
    Batch_data instance(int index);
    
    static int bytes_extra(Hyperparam hyp) {
        return hyp.bytes_total();
    }
    static int bytes_total(Hyperparam hyp) {
        return sizeof(Training_data) + bytes_extra(hyp);
    }
} __attribute__((__packed__));

struct Network_state;

Network_state* network_init(Hyperparam hyp);
void network_free(Network_state* state);
void network_restore(Network_state* state);
void network_save(Network_state* state);

void network_shuffle(Training_data const& from, Training_data* into, int offset = 0, bool silent = false);
void network_load_data(jup_str data_file, Hyperparam hyp, Unique_ptr_free<Training_data>* data_train,
    Unique_ptr_free<Training_data>* data_test);

void network_prepare_data(jup_str graph_file, jup_str data_file, Hyperparam hyp);
void network_train(jup_str data_file);
void network_print_data_info(jup_str data_file);


} /* end of namespace jup */
