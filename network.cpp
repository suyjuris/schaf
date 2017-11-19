
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
    int n = hyp.batch_nodes * hyp.batch_nodes;
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
    
    tensorflow::Operation update_op, save_op, restore_op;
    tensorflow::Output x, y, rate, loss, summary_op, print_op;
    
    std::vector<tensorflow::Output> params;

    int step = 0;

    Timer timer;
    float loss_sum = 0;
    int loss_count = 0;

    UNINITIALIZED( tensorflow::ClientSession session     );
    UNINITIALIZED( tensorflow::EventsWriter event_writer );
};

#undef UNINITIALIZED

Network_state* network_init(Hyperparam hyp) {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    auto state = new Network_state;
    Scope& root = state->root;

    state->hyp = hyp;

    auto shape_scalar = Const<int>(root, 0, {0});

    // Network layout
    auto name = [&root](std::string str) { return root.WithOpName(str); };
    
    auto x = Placeholder(name("x"), DT_FLOAT, Placeholder::Shape({hyp.batch_size, hyp.batch_edges()}));
    auto y = Placeholder(name("y"), DT_FLOAT, Placeholder::Shape({hyp.batch_size}));
    
    auto rate = Variable(name("rate"), {}, DT_FLOAT);

    auto w1 = Variable(name("w1"), {hyp.batch_edges(), hyp.a1_size}, DT_FLOAT);
    auto b1 = Variable(name("b1"), {hyp.a1_size}, DT_FLOAT);
    
    auto w2 = Variable(name("w2"), {hyp.a1_size, hyp.a2_size}, DT_FLOAT);
    auto b2 = Variable(name("b2"), {hyp.a2_size}, DT_FLOAT);

    auto a     = Tanh(name("a"),     BiasAdd(root, MatMul(root, x, w1), b1));
    auto y_out = Tanh(name("y_out"), BiasAdd(root, MatMul(root, a, w2), b2));

    auto loss = L2Loss(name("loss"), Sub(root, Reshape(root, y_out, {hyp.batch_size}), y));

    // Update
    std::vector<Output> grad_vars {w1, b1, w2, b2};
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
    Output s_loss = ScalarSummary(root, std::string {"s_loss"}, loss );
    Output s_y    = HistogramSummary(root, std::string {"s_y"},    y    );
    Output s_yout = HistogramSummary(root, std::string {"s_yout"}, y_out);
    Output s_w1 = HistogramSummary(root, std::string {"s_w1"}, w1);
    Output s_w2 = HistogramSummary(root, std::string {"s_w2"}, w2);
    Output s_b1 = HistogramSummary(root, std::string {"s_b1"}, b1);
    Output s_b2 = HistogramSummary(root, std::string {"s_b2"}, b2);
    Output s_x  = HistogramSummary(root, std::string {"s_x" }, x );
    auto summary_op = MergeSummary(name("run"), {s_loss, s_w1, s_w2, s_b1, s_b2, s_x, s_y, s_yout});

    // Printing
    auto print = root.NewSubScope("print");
    auto print_op = Print(print, Const(print, 0, {0}), OutputList {w1, w2, b1, b2}, Print::Summarize(20));
    
    // Initialize
    auto init = root.NewSubScope("init");
    std::vector<Operation> init_ops;
    auto add_init_op = [&init_ops](Output output) { init_ops.push_back(output.op()); };
    add_init_op( Assign(init, w1, ParameterizedTruncatedNormal(init, {hyp.batch_edges(), hyp.a1_size}, 0.f, .05f, -1.f, 1.f)) );
    add_init_op( Assign(init, w2, ParameterizedTruncatedNormal(init, {hyp.a1_size, hyp.a2_size}, 0.f, .05f, -1.f, 1.f)) );
    add_init_op( Assign(init, b1, Const<float>(init, 0.f, {hyp.a1_size})) );
    add_init_op( Assign(init, b2, Const<float>(init, 0.f, {hyp.a2_size}  )) );
    add_init_op( Assign(init, rate, Const<float>(init, global_options.hyp.learning_rate, {})) );
    auto init_op = NoOp(init.WithControlDependencies(init_ops).WithOpName("run"));

    // Checkpoints
    auto checkpoint = root.NewSubScope("checkpoint");
    std::vector<Output> params_op       { rate,   w1,   b1,   w2,   b2 };
    std::vector<std::string> params_str {"rate", "w1", "b1", "w2", "b2"};
    
    Tensor params_name  {DT_STRING, {(int)params_op.size()}}; 
    Tensor params_slice {DT_STRING, {(int)params_op.size()}}; 
    std::vector<DataType>    params_type  {params_op.size(), DT_FLOAT};
    for (int i = 0; i < (int)params_op.size(); ++i) {
        params_name.flat<std::string>()(i) = params_str[i];
    }
    state->save_op = SaveV2(checkpoint.WithOpName("save"), std::string {global_options.param_out},
        params_name, params_slice, params_op);

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

    new (&state->event_writer) EventsWriter {"/mnt/win_ssd/Philipp10/Dokumente/Uni/Bachelorarbeit/schaf/tf_data/test1"};
    assert( state->event_writer.Init() );
    state->event_writer.WriteEvent(event);

    // Initialize session
    new (&state->session) ClientSession {root};

    state->x    = x;
    state->y    = y;
    state->rate = rate;
    state->loss = loss;
    
    state->summary_op = summary_op;
    state->print_op = print_op;

    state->session.Run({}, {}, {init_op}, nullptr);

    return state;
}

void network_batch(Network_state* state, Batch_data const& data) {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    assert(state);

    auto hyp = state->hyp;
    
    ++state->step;
    
    Tensor data_x {DT_FLOAT, {hyp.batch_size, hyp.batch_edges()}};
    std::memcpy(data_x.flat<float>().data(), data.edge_weights.data(), data.edge_weights.size() * sizeof(float));

    //for (int i = 0; i < 16; ++i)
    //    jdbg >= Array_view<float> {data_x.flat<float>().data()+16*i, 16} ,0;
    
    Tensor data_y {DT_FLOAT, {hyp.batch_size}};
    std::memcpy(data_y.flat<float>().data(), data.results.data(), data.results.size() * sizeof(float));
    
    std::vector<Tensor> outputs;
    TF_CHECK_OK(state->session.Run(
        {{state->x, data_x}, {state->y, data_y}},
        {state->loss, state->summary_op},
        {state->update_op},
        &outputs
    ));

    state->loss_sum += outputs[0].scalar<float>()();
    state->loss_count++;
    
    if (state->timer.update()) {
        jout << "  Loss:  " << jup_printf("%.1f", state->loss_sum / state->loss_count) << " (iteration " << state->step;
        if (global_options.iter_max < std::numeric_limits<int>::max()) {
            jout << "/" << global_options.iter_max;
        }
        jout << ", " << state->timer.counter(state->step) << " iter/s)" << endl;

        state->loss_sum = 0;
        state->loss_count = 0;
    }
    
    Event event;
    event.set_wall_time(elapsed_time());
    event.set_step(state->step);
    event.mutable_summary()->ParseFromString(outputs[1].scalar<std::string>()());
    state->event_writer.WriteEvent(event);
}

/*struct Tensor_data {
    Flat_array64<char> name;
    Flat_array64<int> dims;
    Flat_array64<float> data;
};

struct Checkpoint {
    constexpr static u32 magic = 0xe14a9226;
    
    Hyperparam hyp;
    int iter;
    Flat_array64_const<Tensor_data> tensors;
    };*/

void network_save(Network_state* state) {
    using namespace tensorflow;
    using namespace tensorflow::ops;
    
    assert(state);

    TF_CHECK_OK(state->session.Run({}, {}, {state->save_op}, nullptr));
    
    /*std::vector<Tensor> outputs;
    TF_CHECK_OK(state->session.Run({}, state->params, {}, &outputs));

    Buffer buf;
    
    int size = sizeof(Checkpoint) + outputs.size() * sizeof(Tensor_data);
    for (int i = 0; i < outputs.size(); ++i) {
        size += decltype(Tensor_data{}.name)::extra_space(state->params[i].name().size() + 1);
        size += decltype(Tensor_data{}.dims)::extra_space(outputs[i].rank());
        size += decltype(Tensor_data{}.data)::extra_space(outputs[i].size());
    }

    auto guard = data.reserve_guard(size);
    auto& checkpoint = data.emplace_back<Checkpoint>();
    checkpoint.tensors.init(outputs.size(), &buf);
    
    for (int i = 0; i < outputs.size(); ++i) {
        checkpoints[i].name.init(state->params[i].name().size(), &buf);
        checkpoints[i].dims.init(outputs[i].rank(), &buf);
        checkpoints[i].data.init(outputs[i].size());

        std::memcpy(checkpoints[i].name.begin(), state->params[i].name().begin(), checkpoints.name.size());
        for (int j = 0; j < outputs[i].rank(); ++j) {
            checkpoints[i].dims[j] = outputs[i].dimension(j);
        }
        std::memcpy(checkpoints[i].data.begin(), outputs[i].data(), outputs[i].size());
        }*/
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

struct Node_prio {
    u32 node;
    u32 weight;

    bool operator< (Node_prio o) const {
        return node < o.node;
    }
};

struct Neighbourhood_finder {
    Array<u32> result;
    Array<Node_prio> nodes;

    Array_view<u32> find(Graph const& graph, u32 node, int count);
};

__jup_dbg(Node_prio, weight)

int interpolation_search(Array_view<Node_prio> arr, u32 node) {
    if (arr.size() < 256) {
        int result = 0;
        for (int i = 0; i < arr.size(); ++i) {
            result += arr[i].node < node;
        }
        return (result < arr.size() and arr[result].node == node) ? result : -1;
    }
    
    int beg = 0;
    int end = arr.size() - 1;

    while (beg < end and arr[beg].node <= node and node < arr[end].node) {
        int exp = beg + (node - arr[beg].node) * (end - beg) / (arr[end].node - arr[beg].node);
        if (arr[exp].node < node) {
            beg = exp + 1;
        } else if (arr[exp].node > node) {
            end = exp - 1;
        } else {
            return exp;
        }
    }
    if (arr.size() and arr[end].node == node) {
        return end;
    } else {
        return -1;
    }
}

Array_view<u32> Neighbourhood_finder::find(Graph const& graph, u32 node, int count) {
    nodes.reserve(16);
    nodes.reset();
    result.reset();

    result.push_back(node);

    while (result.size() < count) {
        std::sort(nodes.begin(), nodes.end());

        auto size = nodes.size();
        u32 last = result.back();
        for (Edge i: graph.adjacent(last)) {
            if (result.count(i.other)) continue;

            auto index = interpolation_search({nodes.data(), size}, i.other);
            if (index != -1) {
                nodes[index].weight += i.weight;
            } else {
                if (i.weight > 2 or (int)i.weight > nodes.size()) {
                    nodes.push_back({i.other, i.weight});
                }
            }
        }

        if (not nodes) {
            break;
        }

        auto f = [](u32 x) { return (x-1)*(x-1); };

        int sum = 0;
        for (int i = 0; i < nodes.size(); ++i) {
            sum += f(nodes[i].weight);
            
        }
        
        int arg = -1;
        if (sum == 0) {
            arg = global_rng.gen_uni(nodes.size());
        } else {        
            int val = global_rng.gen_uni(sum);
            for (int i = 0; i < nodes.size(); ++i) {
                val -= f(nodes[i].weight);
                if (val < 0) {
                    arg = i;
                    break;
                }
            }
        }
        assert(arg >= 0);
        
        result.push_back(nodes[arg].node);
        nodes[arg] = nodes.back();
        nodes.addsize(-1);
    }

    while (result.size() < count) {
        result.push_back(-1);
    }
    assert(result.size() == count);

    std::sort(result.begin(), result.end());
    
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
    int num_rewind = 0;

    int const n = data->hyp.batch_nodes;
    int const m = data->hyp.batch_edges();

    while (cur_batch < data->hyp.batch_count) {
        Batch_data out = data->batch(cur_batch);
        
        if (graph_left == 0) {
            bool c = graph_reader_next(&state_graph);
            ++num_graph;
            if (not c) {
                jout << "  Rewinding for more data" << endl;
                graph_reader_reset(&state_graph);
                c = graph_reader_next(&state_graph);
                assert(c);
                ++num_rewind;
            }
                
            graph_left = state_graph.graph->num_nodes() / data->hyp.gen_graph_nodes;
            //jdbg < "Loading graph" < state_graph.graph->name.begin() < graph_left ,0;
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
                std::memset(out.edge_weights.data(), 0, out.edge_weights.size() * sizeof(float));
            }
            
            u32 node = global_rng.gen_uni(state_graph.graph->num_nodes());
            auto nodes = neighbours.find(*state_graph.graph, node, n);

            for (int i = 0; i < nodes.size(); ++i) {
                int j = 0;
                if (nodes[i] == (u32)-1) continue;
                for (Edge e: state_graph.graph->adjacent(nodes[i])) {
                    while (j < nodes.size() and e.other > nodes[j]) ++j;
                    if (j >= nodes.size()) break;
                    if (e.other != nodes[j]) continue;
                    out.edge_weights[cur_instance*m + i*n + j] = e.weight;
                }
            }

            // Whether to have a positive or a negative example
            if (global_rng.gen_bool()) {
                /*
                for (int i = 0; i < 32; ++i) {
                    // Swap edges a and b
                    u32 a = global_rng.gen_uni(m - n);
                    u32 b = global_rng.gen_uni(m - n - 1);
                    b += b >= a;
                    a += a / n + 1;
                    b += b / n + 1;

                    int a_ = (a % n) * n + a / n;
                    int b_ = (b % n) * n + b / n;

                    assert(out.edge_weights[cur_instance*m + a] == out.edge_weights[cur_instance*m + a_]);
                    assert(out.edge_weights[cur_instance*m + b] == out.edge_weights[cur_instance*m + b_]);
                    std::swap(out.edge_weights[cur_instance*m + a ], out.edge_weights[cur_instance*m + b ]);
                    std::swap(out.edge_weights[cur_instance*m + a_], out.edge_weights[cur_instance*m + b_]);
                }
                */
                
                for (int i = 0; i < 32; ++i) {
                    // Add 10 to a random edge
                    u32 a = global_rng.gen_uni(m - n);
                    a += a / n + 1;

                    out.edge_weights[cur_instance*m + a] += 10;
                }

                out.results[cur_instance] = -1.f;
            } else {
                out.results[cur_instance] = 1.f;
            }

            // Normalize
            float max = 1.f;
            for (int i = 0; i < m; ++i) {
                max = std::max(max, out.edge_weights[cur_instance*m + i]);
            }
            //jdbg < max ,0;
            for (float& f: out.edge_weights.subview(cur_instance*m, m)) {
                f /= max;
                //f = std::log(f/max*M_E + exp(-1));
            }
            /*for (int i = 0; i < n; ++i) {
                jdbg >= out.edge_weights.subview(cur_instance*m + i*n, n) ,0;
            }
            jdbg ,0;
            if (cur_instance == 5) std::exit(0);*/

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
         << ", " << num_graph << " graphs, " << num_rewind << " rewinds)" << endl;
}

void network_shuffle(Training_data const& from, Training_data* into) {
    assert(into);
    assert(from.hyp.valid() and into->hyp.valid());

    if (from.hyp.batch_nodes != into->hyp.batch_nodes) {
        die("Incompatible sets of hyperparameters, different number of nodes per instance (have: %d"
            ", want: %d)", from.hyp.batch_nodes, into->hyp.batch_nodes+0);
    } else if (from.hyp.floats_total() < into->hyp.floats_total()) {
        die("Incompatible sets of hyperparameters, there is not enough data to fill the target "
            "(have: %d floats, want: %d floats)", from.hyp.floats_total(), into->hyp.floats_total());
    }
    
    Timer timer;

    Array<int> left;
    left.resize(from.hyp.num_instances());
    for (int i = 0; i < left.size(); ++i) left[i] = i;

    Rng rng;
    for (int i = 0; i < into->hyp.num_instances(); ++i) {
        int index = rng.gen_uni(left.size());
        int j = left[index];
        left[index] = left.pop_back();

        // Move instance j into instance i
        auto j_inst = const_cast<Training_data&>(from).instance(j);
        auto i_inst = into->instance(i);
        jup_memcpy(&i_inst.edge_weights, j_inst.edge_weights);
        jup_memcpy(&i_inst.results, j_inst.results);
    }
    
    jout << "Done. (" << timer.total()  << ", "
         << timer.bytes_done(into->hyp.bytes_total()) << ")" << endl;
}

u64 hash_order_independent(Training_data* data) {
    u64 hash = 0;
    for (int i = 0; i < data->hyp.num_instances(); ++i) {
        auto inst = data->instance(i);
        hash += inst.edge_weights.get_hash() + inst.results.get_hash();
    }
    return hash;
}

void network_prepare_data(jup_str graph_file, jup_str data_file, Hyperparam hyp) {
    // Test whether we can write to the file
    save_bytes(data_file, (u32)0);

    jout << "Preparing data (" << hyp << ")" << endl;
    auto data = Training_data::make_unique(hyp);
    network_generate_data(graph_file, data.get());

    u64 hash_val_oi = hash_order_independent(data.get());
    jout << "Shuffling... (shuffle-invariant checksum: " << nice_hex(hash_val_oi) << ")" << endl;
    auto data2 = Training_data::make_unique(hyp);
    network_shuffle(*data, data2.get());
    assert(hash_val_oi == hash_order_independent(data2.get()));
    std::swap(data, data2);

    u64 hash_val = XXH64(data.get(), Training_data::bytes_total(data->hyp), 0);
    jout << "Writing to file " << data_file << ", Checksum (xxHash64): " << nice_hex(hash_val) << endl;
    save_bytes(data_file, *data, hyp.bytes_total());
}

void network_train(jup_str data_file) {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    auto hyp = global_options.hyp;

    jout << "Hyperparameters:\n"
         << "  batch_count:         " << hyp.batch_count << "\n"
         << "  batch_size:          " << hyp.batch_size << "\n"
         << "  batch_nodes:         " << hyp.batch_nodes << "\n"
         << "  gen_graph_nodes:     " << hyp.gen_graph_nodes << "\n"
         << "  learning_rate:       " << hyp.learning_rate << '\n'
         << "  a1_size:             " << hyp.a1_size << '\n'
         << "  a2_size:             " << hyp.a2_size << '\n';
    
    jout << "Initializing..." << endl;
    jout << "  Loading training data..." << endl;
    
    Hyperparam hyp_td;
    load_bytes(data_file, &hyp_td, -1);
    auto data_orig = Training_data::make_unique(hyp_td);
    load_bytes(data_file, data_orig.get(), Training_data::bytes_extra(hyp_td));
    
    auto data     = Training_data::make_unique(hyp);
    auto data_tmp = Training_data::make_unique(hyp);
    data->hyp     = hyp;
    data_tmp->hyp = hyp;

    jout << "  Shuffling... "; jout.flush();
    network_shuffle(*data_orig, data.get());
    data_orig.release();

    jout << "  Creating neural network..." << endl;
    auto state = network_init(hyp);

    if (global_options.param_in) {
        jout << "  Restoring from checkpoint... (" << global_options.param_in << ")" << endl;
        network_restore(state);
    }

    jout << "Training..." << endl;
    //int initial_step = state->step;
    int cur_batch = 0;
    while (state->step < global_options.iter_max) {
        network_batch(state, data->batch(cur_batch));
        ++cur_batch;
        
        if (cur_batch >= data->hyp.batch_count) {
            jout << "  Re-shuffling training data... "; jout.flush();
            network_shuffle(*data, data_tmp.get());
            std::swap(data, data_tmp);
            cur_batch = 0;
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

    u64 hash_val_oi = hash_order_independent(data.get());
    u64 hash_val = XXH64(data.get(), Training_data::bytes_total(data->hyp), 0);
    
    jout << "\nData:\n";
    jout << "  batch_count:         " << hyp.batch_count << "\n"
         << "  batch_size:          " << hyp.batch_size << "\n"
         << "  batch_nodes:         " << hyp.batch_nodes << "\n"
         << "  gen_graph_nodes:     " << hyp.gen_graph_nodes << "\n";

    jout << "\nStatistics:\n";
    jout << "  total floats:        " << hyp.floats_total() << "\n"
         << "  total bytes:         " << nice_bytes(size_td) << "\n"
         << "  checksum (xxHash64): " << nice_hex(hash_val) << '\n'
         << "  checksum (o.i.):     " << nice_hex(hash_val_oi) << '\n';
}

/*static void network_run_op(Network_state* state, jup_str name) {
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
    
