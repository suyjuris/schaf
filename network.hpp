#pragma once
 
#include "flat_data.hpp"
#include "utilities.hpp"

namespace jup {

#define JUP_DEFAULT_BATCH_COUNT      256
#define JUP_DEFAULT_BATCH_SIZE       256
#define JUP_DEFAULT_RECF_NODES         8
#define JUP_DEFAULT_RECF_COUNT         8
#define JUP_DEFAULT_GEN_GRAPH_NODES   32
#define JUP_DEFAULT_LEARNING_RATE      0.1
#define JUP_DEFAULT_TEST_FRAC          0.1
#define JUP_DEFAULT_DROPOUT            1.0
#define JUP_DEFAULT_L2REG              0.0

struct Hyperparam {
    int batch_count = JUP_DEFAULT_BATCH_COUNT; // batches per training data
    int batch_size  = JUP_DEFAULT_BATCH_SIZE;  // instances per batch
    
    int recf_nodes = JUP_DEFAULT_RECF_NODES; // nodes per receptive field
    int recf_count = JUP_DEFAULT_RECF_COUNT; // number of receptive fields

    int gen_graph_nodes = JUP_DEFAULT_GEN_GRAPH_NODES; // nodes needed per instance

    float learning_rate = (float)JUP_DEFAULT_LEARNING_RATE; // you should know what that means...
    int learning_rate_decay = 0; // number of epochs after which the learning rate is halved.

    float test_frac = (float)JUP_DEFAULT_TEST_FRAC; // amount of training data to use as test data

    int a1_size = 8; // size of the output of the first convolutional layer
    
    int b1_size = 64; // size of the output of the first layer
    int b2_size =  1; // size of the output of the second layer (has to be 1, if it is the last layer)

    float dropout = (float)JUP_DEFAULT_DROPOUT;
    float l2_reg = (float)JUP_DEFAULT_L2REG;

    int num_instances() const {
        return batch_count * batch_size;
    }
    int floats_edge_weights() const {
        return batch_size * recf_count * recf_nodes * recf_nodes;
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
        return batch_count > 0 and batch_size > 0 and recf_nodes > 0;
    }
    int edges_recf() const {
        return recf_nodes * recf_nodes;
    }
    int edges_instance() const {
        return edges_recf() * recf_count;
    }
} __attribute__((__packed__));

struct Batch_data {
    Array_view_mut<float> edge_weights;
    Array_view_mut<float> results;

    float& edge(int instance, int recf, int from, int to, Hyperparam hyp) {
        assert(0 <= instance and instance < hyp.batch_size);
        assert(0 <= recf and recf < hyp.recf_count);
        assert(0 <= from and from < hyp.recf_nodes);
        assert(0 <= to and to < hyp.recf_nodes);
        return edge_weights[instance*hyp.edges_instance() + recf*hyp.edges_recf()
            + from*hyp.recf_nodes + to];
    }
};

struct Training_data {
    union {
        Hyperparam hyp; // This must be in front!
        char _buffer[256];
    };
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
void network_test();
void network_print_data_info(jup_str data_file);


} /* end of namespace jup */
