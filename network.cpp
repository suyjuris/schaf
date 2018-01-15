
#include "libs/sparsehash.hpp"
#include "libs/xxhash.hpp"

#include "array.hpp"
#include "buffer.hpp"
#include "graph.hpp"
#include "options.hpp"
#include "network.hpp"
#include "system.hpp"
#include "utilities.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-compare"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/events_writer.h"

#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"


// tensorflow includes assert.h, which would overwrite our include. We can't have that.
#undef assert
#define assert __jup_assert


#include "debug.hpp"

#pragma GCC diagnostic pop

namespace tensorflow {
namespace ops {
namespace {

Status FloorGrad(const Scope& scope, const Operation& op, const std::vector<Output>& grad_inputs,
        std::vector<Output>* grad_outputs) {
    grad_outputs->push_back(NoGradient());
    return scope.status();
}
REGISTER_GRADIENT_OP("Floor", FloorGrad);

}
} /* end of namespace ops */
} /* end of namespace tensorflow */


namespace jup {

Batch_data Training_data::batch(int index) {
    Batch_data result;
    result.edge_weights = {&batch_data[index * hyp.floats_batch()], hyp.floats_edge_weights()};
    result.results = {result.edge_weights.end(), hyp.floats_results()};
    assert(result.results.end() <= batch_data.end());
    assert(result.results.end() == batch_data.begin() + (index + 1) * hyp.floats_batch());
    return result;
}
Batch_data Training_data::instance(int index) {
    int i_b = index / hyp.batch_size;
    int i_i = index % hyp.batch_size;
    Batch_data b = batch(i_b);
    int n = hyp.edges_instance();
    return {b.edge_weights.subview(i_i * n, n), b.results.subview(i_i, 1)};
}

Unique_ptr_free<Training_data> Training_data::make_unique(Hyperparam hyp) {
    Unique_ptr_free<Training_data> result {(Training_data*)std::calloc(sizeof(Training_data) + hyp.bytes_total(), 1)};
    result->hyp = hyp;
    result->batch_data.m_size = hyp.floats_total();
    return result;
}

#define UNINITIALIZED(x) union {char JUP_UNIQUE_NAME(__dummy) = 0; x;}

struct Network_state {
    Network_state(): root {tensorflow::Scope::NewRootScope()} {}
    ~Network_state() {
        event_writer.~EventsWriter();
        session.~ClientSession();
    }

    Hyperparam hyp;
    
    tensorflow::Scope root;
    
    tensorflow::Operation update_op, save_op, restore_op, decay_op, dropout_on_op, dropout_off_op,
        print_op;
    tensorflow::Output x, y, rate, loss, summary_op;
    
    std::vector<tensorflow::Output> params;

    int step = 0;
    int epoch = 0;
    int epoch_start = 0;

    Timer timer;
    float loss_sum = 0;
    int loss_count = 0;

    UNINITIALIZED( tensorflow::ClientSession session     );
    UNINITIALIZED( tensorflow::EventsWriter event_writer );
};

#undef UNINITIALIZED

static tensorflow::Output Dropout(tensorflow::Scope& scope, tensorflow::Input x, tensorflow::Input p) {
    using namespace tensorflow;
    using namespace tensorflow::ops;
    
    // Similar to the python dropout, see tensorflow/python/ops/nn_ops.py
    auto dropout = scope.NewSubScope("dropout");
    auto binary = Floor(dropout, Add(dropout, RandomUniform(dropout, Shape(dropout, x), DT_FLOAT), p));
    return Mul(dropout, Div(dropout, x, p), binary);
}

Network_state* network_init(Hyperparam hyp) {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    auto state = new Network_state;
    Scope& root = state->root;

    state->hyp = hyp;

    auto shape_scalar = Const<int>(root, 0, {0});
    auto batch_size_inv = Const<float>(root, 1.f / (float)hyp.batch_size);

    // Network layout
    auto name = [&root](std::string str) { return root.WithOpName(str); };
    
    auto x = Placeholder(name("x"), DT_FLOAT, Placeholder::Shape({hyp.batch_size, hyp.recf_count, hyp.edges_recf()}));
    auto y = Placeholder(name("y"), DT_FLOAT, Placeholder::Shape({hyp.batch_size}));
    
    auto rate = Variable(name("rate"), {}, DT_FLOAT);
    auto drop = Variable(name("drop"), {}, DT_FLOAT);

    auto a1w = Variable(name("a1w"), {hyp.edges_recf(), hyp.a1_size}, DT_FLOAT);
    auto a1b = Variable(name("a1b"), {hyp.a1_size}, DT_FLOAT);

    auto a2w = Variable(name("a2w"), {hyp.a1_size, hyp.a2_size}, DT_FLOAT);
    auto a2b = Variable(name("a2b"), {hyp.a2_size}, DT_FLOAT);
    
    auto b1w = Variable(name("b1w"), {hyp.a2_size * hyp.recf_count, hyp.b1_size}, DT_FLOAT);
    auto b1b = Variable(name("b1b"), {hyp.b1_size}, DT_FLOAT);
    
    auto b2w = Variable(name("b2w"), {hyp.b1_size, hyp.b2_size}, DT_FLOAT);
    auto b2b = Variable(name("b2b"), {hyp.b2_size}, DT_FLOAT);
    
    auto b3w = Variable(name("b3w"), {hyp.b2_size, hyp.b3_size}, DT_FLOAT);
    auto b3b = Variable(name("b3b"), {hyp.b3_size}, DT_FLOAT);

    auto a1tmp = Reshape(root, x, {hyp.batch_size * hyp.recf_count, hyp.edges_recf()});
    auto a1out = Dropout(root, Tanh(name("a1out"), BiasAdd(root, MatMul(root, a1tmp, a1w), a1b)), drop);
    auto a2out = Dropout(root, Tanh(name("a2out"), BiasAdd(root, MatMul(root, a1out, a2w), a2b)), drop);

    auto b1tmp = Reshape(root, a2out, {hyp.batch_size, hyp.recf_count * hyp.a2_size});
    auto b1out = Dropout(root, Tanh(name("b1out"), BiasAdd(root, MatMul(root, b1tmp, b1w), b1b)), drop);
    auto b2out = Dropout(root, Tanh(name("b2out"), BiasAdd(root, MatMul(root, b1out, b2w), b2b)), drop);
    auto y_out = Tanh(name("y_out"), BiasAdd(root, MatMul(root, b2out, b3w), b3b));
    
    auto loss = Add(name("loss"),
        Mul(root, L2Loss(root, Sub(root, Reshape(root, y_out, {hyp.batch_size}), y)), batch_size_inv),
        Mul(root, AddN(root, std::initializer_list<Output> {
            L2Loss(root, a1w), L2Loss(root, a2w), L2Loss(root, b1w), L2Loss(root, b2w), L2Loss(root, b3w)
        }), hyp.l2_reg)
    );

    // Update
    std::vector<Output> grad_vars {a1w, a1b, a2w, a2b, b1w, b1b, b2w, b2b, b3w, b3b};
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(root.NewSubScope("grad"), {loss}, grad_vars, &grad_outputs));
    
    auto update = root.NewSubScope("update");
    std::vector<Operation> update_ops;
    for (size_t i = 0; i < grad_vars.size(); ++i) {
        Output output = ApplyGradientDescent(update, grad_vars[i], rate, grad_outputs[i]);
        //Output output = ApplyGradientDescent(update, grad_vars[i], rate,
        //    Print(update, grad_outputs[i], OutputList{Const<int>(update, (int)i), grad_outputs[i]}, Print::Summarize(20)));
        update_ops.push_back(output.op());
    }
    state->update_op = NoOp(update.WithControlDependencies(update_ops).WithOpName("run"));

    // Summaries
    auto summary = root.NewSubScope("summary");
    std::vector<Output> summary_ops;
    auto add_summary = [&summary_ops](Output out) { summary_ops.push_back(out); };
    add_summary( ScalarSummary(   root, std::string {"s_loss"}, loss)  );
    add_summary( ScalarSummary(   root, std::string {"s_rate"}, rate)  );
    add_summary( HistogramSummary(root, std::string {"s_x"},    x)     );
    add_summary( HistogramSummary(root, std::string {"s_y"},    y)     );
    add_summary( HistogramSummary(root, std::string {"s_yout"}, y_out) );
    add_summary( HistogramSummary(root, std::string {"s_a1w"}, a1w)    );
    add_summary( HistogramSummary(root, std::string {"s_a1b"}, a1b)    );
    add_summary( HistogramSummary(root, std::string {"s_a2w"}, a2w)    );
    add_summary( HistogramSummary(root, std::string {"s_a2b"}, a2b)    );
    add_summary( HistogramSummary(root, std::string {"s_b1w"}, b1w)    );
    add_summary( HistogramSummary(root, std::string {"s_b1b"}, b1b)    );
    add_summary( HistogramSummary(root, std::string {"s_b2w"}, b2w)    );
    add_summary( HistogramSummary(root, std::string {"s_b2b"}, b2b)    );
    add_summary( HistogramSummary(root, std::string {"s_b3w"}, b3w)    );
    add_summary( HistogramSummary(root, std::string {"s_b3b"}, b3b)    );
    state->summary_op = MergeSummary(name("run"), summary_ops);

    // Printing
    auto print = root.NewSubScope("print");
    state->print_op = Output {Print(print, Const(print, 0, {0}), OutputList {
        x, y, a1w, a1b, a2w, a2b, b1w, b1b, b2w, b2b, b3w, b3b, a1tmp, a1out, b1tmp, b1out, y_out,
        a1out.op().input(1), b1out.op().input(1)
    }, Print::Summarize(200))}.op();
    
    // Initialize
    auto init = root.NewSubScope("init");
    std::vector<Operation> init_ops;
    auto add_init_op = [&init_ops](Output output) { init_ops.push_back(output.op()); };
    for (auto i: {a1w, a2w, b1w, b2w, b3w}) {
        add_init_op( Assign(init, i, ParameterizedTruncatedNormal(init, Shape(init, i), 0.f, .05f, -1.f, 1.f)) );
    }
    for (auto i: {a1b, a2b, b1b, b2b, b3b}) {
        add_init_op( Assign(init, i, Fill(init, Shape(init, i), 0.f)) );
    }
    add_init_op( Assign(init, rate, Const<float>(init, global_options.hyp.learning_rate, {})) );
    add_init_op( Assign(init, drop, Const<float>(init, global_options.hyp.dropout, {})) );
    auto init_op = NoOp(init.WithControlDependencies(init_ops).WithOpName("run"));

    // Dropout initialization
    auto dropout = root.NewSubScope("dropout_set");
    state->dropout_on_op  = Output { Assign(dropout, drop, Const<float>(dropout, global_options.hyp.dropout, {})) }.op();
    state->dropout_off_op = Output { Assign(dropout, drop, Const<float>(dropout, 1.f, {})) }.op();

    // Learning rate decay
    auto decay = root.NewSubScope("decay");
    state->decay_op = Output {Assign(decay, rate, Mul(decay, rate, 0.5f))}.op();

    // Checkpoints
    auto checkpoint = root.NewSubScope("checkpoint");
    std::vector<Output> params_op       { rate,   a1w,   a1b,   a2w,   a2b,   b1w,   b1b,   b2w,   b2b,   b3w,   b3b };
    std::vector<std::string> params_str {"rate", "a1w", "a1b", "a2w", "a2b", "b1w", "b1b", "b2w", "b2b", "b3w", "b3b"};

    Tensor params_name  {DT_STRING, {(int)params_op.size()}}; 
    Tensor params_slice {DT_STRING, {(int)params_op.size()}}; 
    std::vector<DataType> params_type {params_op.size(), DT_FLOAT};
    for (int i = 0; i < (int)params_op.size(); ++i) {
        params_name.flat<std::string>()(i) = params_str[i];
    }
    if (global_options.logdir and global_options.iter_save) {
        std__filesystem::path path {global_options.logdir.c_str()};
        path /= get_date_string().c_str();
        std::error_code ec;
        std__filesystem::create_directories(path, ec);
        if (ec) {
            die("%s\nwhile trying to create directory %s", ec.message(), path.c_str());
        }

        state->save_op = SaveV2(checkpoint.WithOpName("save"), (path / "param").native(), params_name,
            params_slice, params_op);
    }

    auto restore_op_main = RestoreV2(checkpoint, std::string {global_options.param_in},
        params_name, params_slice, params_type);
    std::vector<Operation> restore_ops;
    for (int i = 0; i < (int)params_op.size(); ++i) {
        restore_ops.push_back( Output {Assign(checkpoint, params_op[i], restore_op_main[i])}.op() );
    }
    state->restore_op = NoOp(init.WithControlDependencies(restore_ops).WithOpName("restore"));
    
    // Write graph information
    using google::protobuf::TextFormat;
    using google::protobuf::io::OstreamOutputStream;

    GraphDef graph_def;
    TF_CHECK_OK( root.ToGraphDef(&graph_def) );
    std::string graph_data;
    graph_def.SerializeToString(&graph_data);

    Event event;
    event.set_wall_time(elapsed_time());
    event.set_step(state->step);
    event.set_graph_def(graph_data);

    if (global_options.logdir and global_options.iter_event) {
        std__filesystem::path path {global_options.logdir.c_str()};
        path /= get_date_string().c_str();
        std::error_code ec;
        std__filesystem::create_directories(path, ec);
        if (ec) {
            die("%s\nwhile trying to create directory %s", ec.message(), path.c_str());
        }

        new (&state->event_writer) EventsWriter {(path / "run").native()};
        assert( state->event_writer.Init() );
        state->event_writer.WriteEvent(event);
    }

    SessionOptions session_options;
    //session_options.config.set_log_device_placement(true);
    
    // Initialize session
    new (&state->session) ClientSession {root, session_options};

    state->x    = x;
    state->y    = y;
    state->rate = rate;
    state->loss = loss;
    
    state->session.Run({}, {}, {init_op}, nullptr);

    return state;
}

static void fill_tensors(Hyperparam hyp, Batch_data const& batch, tensorflow::Tensor* data_x, tensorflow::Tensor* data_y) {
    using namespace tensorflow;
    
    if (data_x->NumElements() == 0) {
        *data_x = Tensor {DT_FLOAT, {hyp.batch_size, hyp.recf_count, hyp.edges_recf()}};
    }
    if (data_y->NumElements() == 0) {
        *data_y = Tensor {DT_FLOAT, {hyp.batch_size}};
    }
    
    assert(batch.edge_weights.size() == data_x->NumElements());
    std::memcpy(data_x->flat<float>().data(), batch.edge_weights.data(), batch.edge_weights.size() * sizeof(float));

    
    assert(batch.results.size() == data_y->NumElements());
    std::memcpy(data_y->flat<float>().data(), batch.results.data(), batch.results.size() * sizeof(float));
}

void network_batch(Network_state* state, Batch_data const& data) {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    assert(state);

    auto hyp = state->hyp;

    Tensor data_x, data_y;
    fill_tensors(hyp, data, &data_x, &data_y);

    if (hyp.learning_rate_decay
        and state->epoch % hyp.learning_rate_decay == 0
        and state->epoch_start == state->step
        and state->step > 0
    ) {
        std::vector<Tensor> outputs;
        TF_CHECK_OK(state->session.Run({}, {}, {state->decay_op}, nullptr));
        TF_CHECK_OK(state->session.Run({}, {state->rate}, {}, &outputs));
        jout << "  Reducing learning rate, down to " << outputs[0].scalar<float>()() << "\n";
    }
    
    ++state->step;
    
    std::vector<Tensor> outputs;
    TF_CHECK_OK(state->session.Run(
        {{state->x, data_x}, {state->y, data_y}},
        {state->loss},
        {state->update_op},
        &outputs
    ));

    state->loss_sum += outputs[0].scalar<float>()();
    state->loss_count++;
    
    if (global_options.iter_event and state->step % global_options.iter_event == 0) {
        outputs.clear();
        TF_CHECK_OK(state->session.Run(
            {{state->x, data_x}, {state->y, data_y}},
            {state->summary_op},
            {},
            &outputs
        ));
        
        Event event;
        event.set_wall_time(elapsed_time());
        event.set_step(state->step);
        event.mutable_summary()->ParseFromString(outputs[0].scalar<std::string>()());
        state->event_writer.WriteEvent(event);
    }
}

float network_validate(Network_state* state, Training_data /*const*/& data_test) {
    using namespace tensorflow;
    
    auto hyp = data_test.hyp;

    // Need to turn off dropout during validation
    TF_CHECK_OK(state->session.Run({}, {}, {state->dropout_off_op}, nullptr));
    
    float loss_sum = 0.f;
    Tensor data_x, data_y;
    
    std::vector<Tensor> outputs;

    for (int i = 0; i < hyp.batch_count; ++i) {
        fill_tensors(hyp, data_test.batch(i), &data_x, &data_y);
        TF_CHECK_OK(state->session.Run(
            {{state->x, data_x}, {state->y, data_y}},
            {state->loss},
            {},
            &outputs
        ));
        loss_sum += outputs[0].scalar<float>()();
    }
    
    TF_CHECK_OK(state->session.Run({}, {}, {state->dropout_on_op}, nullptr));

    return loss_sum / hyp.batch_count;
}

void network_save(Network_state* state) {
    using namespace tensorflow;
    using namespace tensorflow::ops;
    
    assert(state);

    // This is a precondition of calling this function
    assert(global_options.logdir and global_options.iter_save);
        
    TF_CHECK_OK(state->session.Run({}, {}, {state->save_op}, nullptr));
}

void network_restore(Network_state* state) {
    using namespace tensorflow;
    using namespace tensorflow::ops;
    
    assert(state);

    TF_CHECK_OK(state->session.Run({}, {}, {state->restore_op}, nullptr));
}

void network_free(Network_state* state) {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    assert(state);

    delete state;
}

struct Memberhip_tester {
    static constexpr int size = 128;
    u64 data[size];

    void clear() {
        std::memset(data, 0, sizeof(u64)*size);
    }
    void add(u32 node) {
        u64 i1 = (node >> 6) % size;
        u64 i2 = node & 0x3f;
        data[i1] |= 1 << i2;
    }
    bool count(u32 node) {
        u64 i1 = (node >> 6) % size;
        u64 i2 = node & 0x3f;
        return data[i1] & (1 << i2);
    }
};

struct Neighbourhood_finder {
    Array<u32> result;
    google::dense_hash_map<u32, u32> nodes_h;
    Memberhip_tester memtest;

    template <bool is_neg>
    Array_view<u32> find(Graph const& graph, u32 node, int count);

    Neighbourhood_finder() {
        nodes_h.set_empty_key((u32)-1);
        nodes_h.set_deleted_key((u32)-2);
    }
};

static u32 perturb(u32 a, u32 b, u32 edge) {
    // splitmix64, see http://xorshift.di.unimi.it/splitmix64.c
    u64 val = a < b ? (u64)a << 32 | (u64)b : (u64)b << 32 | (u64)a;
    val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9ull;
    val = (val ^ (val >> 27)) * 0x94d049bb133111ebull;
    val = val ^ (val >> 31);

    u32 perturb = (val & 0xff) < 16 ? __builtin_clz((val >> 8) | 1) : 0;
    return edge + perturb;
}

template <bool is_neg>
Array_view<u32> Neighbourhood_finder::find(Graph const& graph, u32 node, int count) {
    nodes_h.clear();
    memtest.clear();
    
    result.reset();
    result.push_back(node);
    memtest.add(node);

    while (result.size() < count) {
        u32 last = result.back();
        for (Edge i: graph.adjacent(last)) {
            if (memtest.count(i.other) and result.count(i.other)) continue;
            
            nodes_h[i.other] += is_neg ? perturb(last, i.other, i.weight) : i.weight;
        }

        if (nodes_h.size() == 0) break;

        auto f = [](u32 x) { return (x-1)*(x-1); };

        int sum = 0;
        for (auto i: nodes_h) {
            sum += f(i.second);
        }
        
        u32 arg_node = -1;
        if (sum == 0) {
            int val = global_rng.gen_uni(nodes_h.size());
            for (auto i: nodes_h) {
                --val;
                if (val < 0) {
                    arg_node = i.first;
                    break;
                }
            }
        } else {
            int val = global_rng.gen_uni(sum);
            for (auto i: nodes_h) {
                val -= f(i.second);
                if (val < 0) {
                    arg_node = i.first;
                    break;
                }
            }
        }
        assert(arg_node != (u32)-1);
        
        result.push_back(arg_node);
        memtest.add(arg_node);
        nodes_h.erase(arg_node);
    }
    
    while (result.size() < count) {
        result.push_back(-1);
    }
    assert(result.size() == count);
    
    return result;
}

void network_generate_data(jup_str graph_file, Training_data* data) {
    assert(data);
    assert(data->hyp.valid());

    jout << "Generating training data... ("
         << nice_bytes(data->hyp.bytes_total()) << ")" << endl;
    Timer timer {(u64)(data->hyp.batch_count * data->hyp.batch_size)};

    Graph_reader_state state_graph;
    graph_reader_init(&state_graph, graph_file);

    Neighbourhood_finder neighbours;

    int graph_left = 0;
    int cur_batch = 0;
    int cur_instance = 0;

    int num_graph = 0;
    int num_random = 0;
    int num_rewind = 0;

    bool cur_graph_random;

    while (cur_batch < data->hyp.batch_count) {
        Batch_data out = data->batch(cur_batch);
        
        if (graph_left == 0) {
            cur_graph_random = global_rng.gen_bool(256 / 4);
            if (cur_graph_random) {
                graph_reader_random(&state_graph, &global_rng);
                ++num_random;
            } else {
                bool c = graph_reader_next(&state_graph);
                ++num_graph;
                if (not c) {
                    jout << "  Rewinding for more data" << endl;
                    graph_reader_reset(&state_graph);
                    c = graph_reader_next(&state_graph);
                    assert(c);
                    ++num_rewind;
                }    
            }
                
            graph_left = state_graph.graph->num_nodes() / data->hyp.gen_graph_nodes;
            if (graph_left == 0) continue;
        }

        if (timer.update()) {
            auto s = jup_printf(" (batch %d/%d, instance %d/%d, graph %s, ", cur_batch, data->hyp.batch_count,
                cur_instance, data->hyp.batch_size, state_graph.graph->name.begin());
            jout << "  Currently at " << timer.progress(cur_batch * data->hyp.batch_size + cur_instance) << s
                 << timer.bytes(cur_batch * data->hyp.bytes_batch() + cur_instance * data->hyp.bytes_instance())
                 << ")" << endl;
        }

        while (cur_instance < data->hyp.batch_size) {
            if (cur_instance == 0) {
                jup_memset(&out.edge_weights);
            }

            bool do_falsify = not cur_graph_random and global_rng.gen_bool(256 / 3);
            
            for (int cur_recf = 0; cur_recf < data->hyp.recf_count; ++cur_recf) {
                u32 node = global_rng.gen_uni(state_graph.graph->num_nodes());
                auto nodes = do_falsify ?
                    neighbours.find<true >(*state_graph.graph, node, data->hyp.recf_nodes):
                    neighbours.find<false>(*state_graph.graph, node, data->hyp.recf_nodes);

                // Ignore empty slots at the end
                int nodes_end = nodes.size();
                while (nodes_end > 0 and nodes[nodes_end - 1] == (u32)-1) --nodes_end;

                // Fill in the edges
                for (int i = 0; i+1 < nodes_end; ++i) {
                    for (Edge e: state_graph.graph->adjacent(nodes[i])) {
                        for (int j = i+1; j < nodes_end; ++j) {
                            if (e.other != nodes[j]) continue;
                            u32 weight = do_falsify ? perturb(nodes[i], nodes[j], e.weight) : e.weight;
                            out.edge(cur_instance, cur_recf, i, j, data->hyp) = weight;
                            out.edge(cur_instance, cur_recf, j, i, data->hyp) = weight;
                        }
                    }
                }
            }

            // Whether to have a positive or a negative example
            if (do_falsify or cur_graph_random) {
                out.results[cur_instance] = -1.f;
            } else {
                out.results[cur_instance] = 1.f;
            }

            // Normalize
            int m = data->hyp.edges_instance();
            float max = 1.f;
            for (float f: out.edge_weights.subview(cur_instance*m, m)) {
                if (f > max) max = f;
            }

            // Empirical values for normalization
            // cdf = c2 * exp(c1*x) + c3
            constexpr float c1 = 4.2991;
            constexpr float c2 = 0.0154927;
            constexpr float c3 = -0.0184841;
            
            for (float& f: out.edge_weights.subview(cur_instance*m, m)) {
                f = std::log((f/max - c3)/c2) / c1 * 2.f - 1.f;
            }

            ++cur_instance;
            --graph_left;

            if (graph_left == 0) break;
        }

        
        if (cur_instance == data->hyp.batch_size) {
            cur_instance = 0;
            ++cur_batch;
        }
    }

    graph_reader_close(&state_graph);

    jout << "Done. (" << timer.total() << ", " << timer.bytes_done(cur_batch * data->hyp.bytes_batch())
         << ", " << num_graph << " ordinary graphs, " << num_random << " random graphs, " << num_rewind
         << " rewinds)" << endl;
}

void network_shuffle(Training_data const& from, Training_data* into, int offset, bool silent) {
    assert(into);
    assert(from.hyp.valid() and into->hyp.valid());
    assert(0 <= offset and offset < from.hyp.batch_count);

    if (from.hyp.recf_nodes != into->hyp.recf_nodes) {
        die("Incompatible sets of hyperparameters, different number of nodes per instance (have: %d"
            ", want: %d)", from.hyp.recf_nodes, into->hyp.recf_nodes+0);
    } else if (from.hyp.floats_batch() * (from.hyp.batch_count - offset) < into->hyp.floats_total()) {
        die("Incompatible sets of hyperparameters, there is not enough data to fill the target "
            "(have: %d floats, want: %d floats)", from.hyp.floats_total(), into->hyp.floats_total());
    }
    
    Timer timer;

    Array<int> left;
    left.resize(into->hyp.num_instances());
    for (int i = 0; i < left.size(); ++i) left[i] = i + offset * from.hyp.batch_size;

    Rng rng;
    for (int i = 0; i < into->hyp.num_instances(); ++i) {
        int index = rng.gen_uni(left.size());
        int j = left[index];
        left[index] = left.back();
        left.pop_back();

        // Move instance j into instance i
        if (j >= from.hyp.num_instances()) {
            jdbg < j < from.hyp.num_instances() < index ,0;
        }
        auto j_inst = const_cast<Training_data&>(from).instance(j);
        auto i_inst = into->instance(i);
        jup_memcpy(&i_inst.edge_weights, j_inst.edge_weights);
        jup_memcpy(&i_inst.results, j_inst.results);
    }

    if (not silent) {
        jout << "Done. (" << timer.total()  << ", "
             << timer.bytes_done(into->hyp.bytes_total()) << ")" << endl;
    }
}

static u64 hash_shuffle_inv(Training_data /*const*/& data, int last_inst = -1) {
    assert(last_inst <= data.hyp.num_instances());
    u64 hash = 0;
    int end = last_inst < 0 ? data.hyp.num_instances() : last_inst;
    for (int i = 0; i < end; ++i) {
        auto inst = data.instance(i);
        hash += inst.edge_weights.get_hash() + inst.results.get_hash();
    }
    return hash;
}

static void print_hyperparam(Hyperparam hyp) {
    jout << "Hyperparameters:\n"
         << "  batch_count:         " << hyp.batch_count << "\n"
         << "  batch_size:          " << hyp.batch_size << "\n"
         << "  recf_nodes:          " << hyp.recf_nodes << "\n"
         << "  recf_count:          " << hyp.recf_count << "\n"
         << "  gen_graph_nodes:     " << hyp.gen_graph_nodes << "\n"
         << "  learning_rate:       " << hyp.learning_rate << '\n'
         << "  learning_rate_decay: " << hyp.learning_rate_decay << '\n'
         << "  test_frac:           " << hyp.test_frac << '\n'
         << "  a1_size:             " << hyp.a1_size << '\n'
         << "  a2_size:             " << hyp.a2_size << '\n'
         << "  b1_size:             " << hyp.b1_size << '\n'
         << "  b2_size:             " << hyp.b2_size << '\n'
         << "  b3_size:             " << hyp.b3_size << '\n'
         << "  dropout:             " << hyp.dropout << '\n'
         << "  l2_reg:              " << hyp.l2_reg << '\n';
}

void network_prepare_data(jup_str graph_file, jup_str data_file, Hyperparam hyp) {
    // Test whether we can write to the file
    save_bytes(data_file, (u32)0);

    jout << "Preparing data...\n";
    print_hyperparam(hyp);
    auto data = Training_data::make_unique(hyp);
    network_generate_data(graph_file, data.get());

    u64 hash_val_oi = hash_shuffle_inv(*data);
    jout << "Shuffle-invariant checksum: " << nice_hex(hash_val_oi) << endl;

    // Do not shuffle to have more variance between test and training data later
    /*u64 hash_val_oi = hash_shuffle_inv(*data);
    jout << "Shuffling... (shuffle-invariant checksum: " << nice_hex(hash_val_oi) << ")" << endl;
    auto data2 = Training_data::make_unique(hyp);
    network_shuffle(*data, data2.get());
    assert(hash_val_oi == hash_shuffle_inv(*data2));
    std::swap(data, data2);*/

    u64 hash_val = XXH64(data.get(), Training_data::bytes_total(data->hyp), 0);
    jout << "Writing to file " << data_file << ", Checksum (xxHash64): " << nice_hex(hash_val) << endl;
    save_bytes(data_file, *data, hyp.bytes_total());
}

void network_load_data(
    jup_str data_file, Hyperparam hyp, Unique_ptr_free<Training_data>* data_train,
    Unique_ptr_free<Training_data>* data_test
) {
    assert(data_train and data_test);

    jout << "  Loading training data..." << endl;

    // Load the file
    Hyperparam hyp_td;
    load_bytes(data_file, &hyp_td, -1);
    auto data_orig = Training_data::make_unique(hyp_td);
    load_bytes(data_file, data_orig.get(), Training_data::bytes_extra(hyp_td));

    // Check that there is enough data
    if (hyp_td.floats_total() < hyp.floats_total()) {
        die("Set of training data is too small, there is not enough data to fill the target "
            "(have: %d floats, want: %d floats)", hyp_td.floats_total(), hyp.floats_total());        
    }

    Hyperparam hyp_train = hyp;
    Hyperparam hyp_test  = hyp;
    hyp_train.batch_count = (1.f - hyp.test_frac) * hyp.batch_count;
    hyp_test.batch_count = hyp.batch_count - hyp_train.batch_count;
    jout << "  Using " << hyp_train.batch_count << " batches for training, " << hyp_test.batch_count
         << " for testing\n";
    
    *data_train = Training_data::make_unique(hyp_train);
    *data_test  = Training_data::make_unique(hyp_test );

    // Do the copying via shuffle (which also checks that the sizes are correct and handles
    // different values of batch_size correctly
    network_shuffle(*data_orig, data_train->get(), 0, true);
    network_shuffle(*data_orig, data_test->get(), (**data_train).hyp.batch_count, true);

    u64 hash = hash_shuffle_inv(*data_orig, hyp.num_instances());
    assert(hash == hash_shuffle_inv(**data_train) + hash_shuffle_inv(**data_test));
}

void network_train(jup_str data_file) {
    auto hyp = global_options.hyp;
    print_hyperparam(hyp);

    jout << "Initializing..." << endl;

    jout << "  Creating neural network..." << endl;
    auto state = network_init(hyp);

    Unique_ptr_free<Training_data> data_train, data_test, data_tmp;
    network_load_data(data_file, hyp, &data_train, &data_test);
    data_tmp = Training_data::make_unique(data_train->hyp);

    if (global_options.param_in) {
        jout << "  Restoring from checkpoint... (" << global_options.param_in << ")" << endl;
        network_restore(state);
    }

    jout << "Training..." << endl;
    //int initial_step = state->step;
    int cur_batch = 0;
    while (state->step < global_options.iter_max) {
        network_batch(state, data_train->batch(cur_batch));
        ++cur_batch;
        
        if (cur_batch >= data_train->hyp.batch_count) {
            network_shuffle(*data_train, data_tmp.get(), 0, true);
            std::swap(data_train, data_tmp);
            cur_batch = 0;
            ++state->epoch;
            state->epoch_start = state->step;
        }

        if (state->step
            and global_options.iter_save
            and global_options.logdir
            and state->step % global_options.iter_save == 0
        ) {
            network_save(state);
        }

        if (state->timer.update()) {
            float loss_test = network_validate(state, *data_test);
            jout << jup_printf("  Loss: %.4f train, %.4f test", state->loss_sum / state->loss_count, loss_test)
                 << " (epoch " << state->epoch << ", iteration " << state->step;
            if (global_options.iter_max < std::numeric_limits<int>::max()) {
                jout << "/" << global_options.iter_max;
            }
            jout << ", " << state->timer.counter(state->step) << " iter/s)" << endl;

            state->loss_sum = 0;
            state->loss_count = 0;
        }
    }

    network_free(state);
}

void network_print_data_info(jup_str data_file) {
    jout << "Loading file " << data_file << "..." << endl;
    
    Hyperparam hyp;
    load_bytes(data_file, &hyp, -1);

    u64 size = get_file_size(data_file);
    int size_td = Training_data::bytes_total(hyp);
    if ((u64)size_td != size) {
        die("Invalid file size. Expected %d bytes, got %" PRId64 ".", size_td, size);
    }
    
    auto data = Training_data::make_unique(hyp);
    load_bytes(data_file, data.get(), Training_data::bytes_extra(hyp));

    u64 hash_val_oi = hash_shuffle_inv(*data);
    u64 hash_val = XXH64(data.get(), Training_data::bytes_total(data->hyp), 0);

    jout << "\n";
    print_hyperparam(hyp);

    jout << "\nStatistics:\n";
    jout << "  total floats:        " << hyp.floats_total() << "\n"
         << "  total bytes:         " << nice_bytes(size_td) << "\n"
         << "  checksum (xxHash64): " << nice_hex(hash_val) << '\n'
         << "  checksum (s.i.):     " << nice_hex(hash_val_oi) << '\n';
}

void network_random_hyp(Hyperparam* hyp, Rng* rng) {
    int inst_total = hyp->num_instances();
    hyp->batch_size = rng->choose_uni({64, 128, 256, 512});
    hyp->batch_count = inst_total / hyp->batch_size;
    
    hyp->learning_rate = rng->gen_normal(-5.9, 1.0);
    
    hyp->a1_size = rng->choose_uni({4, 8, 16});
    hyp->a2_size = rng->choose_uni({0, 4, 8});
    hyp->b1_size = rng->choose_uni({32, 64, 128});
    hyp->b2_size = rng->choose_uni({1, 16, 32});

    hyp->dropout = rng->choose_uni({0.5, 0.7, 1.0});
    hyp->l2_reg = rng->choose_uni({0.0, 1e-6, 1e-5});
}

void network_grid_search(jup_str data_file) {
    auto hyp = global_options.hyp;
    print_hyperparam(hyp);

    Rng rng {global_rng.rand()};
    
    jout << "Initializing..." << endl;

    setenv("TF_CPP_MIN_LOG_LEVEL", "1", 0);
    
    Unique_ptr_free<Training_data> data_train, data_test, data_train_tmp, data_test_tmp;
    network_load_data(data_file, hyp, &data_train, &data_test);
    data_train_tmp = Training_data::make_unique(data_train->hyp);
    data_test_tmp  = Training_data::make_unique(data_test ->hyp);

    jout << '\n';
    jout << "  loss | trai | iter | batch |  rate  | a1 | a2 | b1 | b2 | dropout | l2\n";
    jout.flush();

    double best_loss = 9e9;
    
    while (true) {
        Hyperparam hyp_rand = data_train->hyp;
        network_random_hyp(&hyp_rand, &rng);

        data_train_tmp->hyp = hyp_rand;
        data_test_tmp ->hyp = hyp_rand;
        data_test_tmp->hyp.batch_count = data_test->hyp.num_instances() / hyp_rand.batch_size;
        
        network_shuffle(*data_train, data_train_tmp.get(), 0, true);
        network_shuffle(*data_test,  data_test_tmp .get(), 0, true);
        
        auto state = network_init(hyp_rand);
        double start_time = elapsed_time();
        int cur_batch = 0;

        while (elapsed_time() < start_time + global_options.grid_max_time) {
            network_batch(state, data_train_tmp->batch(cur_batch));
            ++cur_batch;
        
            if (cur_batch >= data_train_tmp->hyp.batch_count) {
                network_shuffle(*data_train, data_train_tmp.get(), 0, true);
                cur_batch = 0;
                ++state->epoch;
                state->epoch_start = state->step;
            }
        }

        double loss_test  = network_validate(state, *data_test_tmp);
        double loss_train = state->loss_sum / state->loss_count;

        if (loss_test < best_loss) {
            best_loss = loss_test;
            jout << "* ";
        } else {
            jout << "  ";
        }
        
        jout << jup_printf(
            "%5.3f  %5.3f %6d %7d %8.2e %4d %4d %4d %4d  %8.2e %8.2e",
            loss_test, loss_train, state->step, hyp_rand.batch_size, (double)hyp_rand.learning_rate,
            hyp_rand.a1_size, hyp_rand.a2_size, hyp_rand.b1_size, hyp_rand.b2_size,
            (double)hyp_rand.dropout, (double)hyp_rand.l2_reg
        ) << endl;

        network_free(state);
    }
}

/*
void network_test() {
    using namespace tensorflow;
    
    Hyperparam hyp;
    hyp.batch_count = 1;
    hyp.batch_size = 2;
    hyp.recf_nodes = 2;
    hyp.recf_count = 3;
    hyp.a1_size = 3;
    hyp.b1_size = 2;
    hyp.dropout = 0.5;

    auto data = Training_data::make_unique(hyp);
    for (float& i: data->batch(0).edge_weights) {
        i = ((float)global_rng.gen_uni(5) - 2.f) / 2.f;
    }
    for (float& i: data->batch(0).results) {
        i = global_rng.gen_bool() ? -1.f : 1.f;;
    }
    
    auto state = network_init(hyp);

    Tensor data_x, data_y;
    fill_tensors(hyp, data->batch(0), &data_x, &data_y);

    TF_CHECK_OK(state->session.Run(
        {{state->x, data_x}, {state->y, data_y}},
        {},
        {state->print_op},
        nullptr
    ));
    TF_CHECK_OK(state->session.Run(
        {{state->x, data_x}, {state->y, data_y}},
        {},
        {state->print_op},
        nullptr
    ));
    
    network_free(state);
}

static void network_run_op(Network_state* state, jup_str name) {
    using namespace tensorflow;
    using namespace tensorflow::ops;
    
    Node* node = nullptr;
    for (int i = 0; i < state->graph.node_size(); ++i) {
        if (jup_str {state->graph.node(i).name()} == name) {
            //node = state->graph.node(i);
            break;
        }
    }
    assert(node);

    TF_CHECK_OK( state->session.Run({Output {node}}, nullptr) );
    }*/

    /*
    MetaGraphDef meta_graph_def;
    MetaGraphDef::MetaInfoDef meta_info_def;
    meta_info_def.set_tensorflow_version(tf_compiler_version());
    meta_info_def.set_tensorflow_git_version(tf_git_version());
    meta_graph_def.mutable_meta_info_def()->MergeFrom(meta_info_def);

    meta_graph_def.mutable_graph_def()->MergeFrom(graph_def);

    OpList stripped_op_list;
    StrippedOpListForGraph(graph_def, *OpRegistry::Global(), &stripped_op_list);
    meta_graph_def.mutable_meta_info_def()->mutable_stripped_op_list()->MergeFrom(stripped_op_list);
    */

} /* end of namespace jup */

#if 0
    GraphDef graph_def;
    TF_CHECK_OK( state->root.ToGraphDef(&graph_def) );
    Session* session;
    SessionOptions options;
    TF_CHECK_OK( NewSession(options, &session) );
    TF_CHECK_OK( session->Create(graph_def) );

    Tensor data_rate {DT_FLOAT, {}};
    data_rate.scalar<float>()() = 0.01f;
    TF_CHECK_OK(session->Run(
        {},
        {},
        {"init/run"},
        &outputs
    ));
    std::vector<std::string> node_names;
    for (int i = 0; i < graph_def.node_size(); ++i) {
        OpDef const* op_def;
        TF_CHECK_OK( OpRegistry::Global()->LookUpOpDef(graph_def.node(i).op(), &op_def) );
        if (op_def->name() == "Placeholder") continue;
        int output_size = op_def->output_arg_size();
        for (int j = 0; j < output_size; ++j) {
            node_names.push_back(graph_def.node(i).name() + jup_printf(":%d", j).c_str());
        }
    }
    jdbg >= node_names ,0;
    
    outputs.clear();
    TF_CHECK_OK(session->Run(
        {{"x", data_x}, {"y", data_y}, {"rate", data_rate}},
        node_names,
        {},
        &outputs
    ));

    node_names.push_back("x");
    node_names.push_back("y");
    outputs.push_back(data_x);
    outputs.push_back(data_y);
    size_t id3 = 0;
    size_t mat_mul3 = 0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].dtype() != DT_FLOAT) continue;
        if (node_names[i] == "grad/Identity_2:0") id3 = i;
        if (node_names[i] == "grad/MatMul_3:0") mat_mul3 = i;
        jdbg < node_names[i] < outputs[i].flat<float>().size() >= Array_view<float> {outputs[i].flat<float>().data(),
                std::min((int)outputs[i].flat<float>().size(), 100)} ,0;
    }

    
    {auto mat = data_x.matrix<float>();
        /*for (int i = 0; i < mat.dimension(0); ++i) {
        jdbg >= Array_view<float> {mat.data() + i*mat.dimension(1), (int)mat.dimension(1)} ,0;;
        }*/
    for (int i = 0; i < mat.dimension(1); ++i) {
        for (int j = 0; j < mat.dimension(0); ++j) {
            jdbg < mat(j, i);
        }
        jdbg ,0;
    }
    jdbg ,4;}

    {auto mat = outputs[id3].matrix<float>();
    
    for (int i = 0; i < mat.dimension(1); ++i) {
        for (int j = 0; j < mat.dimension(0); ++j) {
            jdbg < mat(j, i);
        }
        jdbg ,0;
    }
    /*for (int i = 0; i < mat.dimension(0); ++i) {
        jdbg >= Array_view<float> {mat.data() + i*mat.dimension(1), (int)mat.dimension(1)} ,0;;
        }*/
    jdbg ,4;}
    {auto mat = outputs[mat_mul3].matrix<float>();
    
    for (int i = 0; i < mat.dimension(1); ++i) {
        for (int j = 0; j < mat.dimension(0); ++j) {
            jdbg < mat(j, i);
        }
        jdbg ,0;
    }
    /*for (int i = 0; i < mat.dimension(0); ++i) {
        jdbg >= Array_view<float> {mat.data() + i*mat.dimension(1), (int)mat.dimension(1)} ,0;;
        }*/
    jdbg ,4;}

    std::exit(0);
#endif
    
