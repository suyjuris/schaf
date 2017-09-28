
#include "array.hpp"
#include "buffer.hpp"
#include "graph.hpp"
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

#define UNINITIALIZED(x) union {char JUP_UNIQUE_NAME(__dummy) = 0; x;}

struct Network_state {
    Network_state(): root {tensorflow::Scope::NewRootScope()} {}
    ~Network_state() {
        event_writer.~EventsWriter();
        session.~ClientSession();
    }
    
    tensorflow::Scope root;
    
    tensorflow::Operation update_op;
    tensorflow::Output x, y, rate, loss, summary_op, print_op;

    int step = 0;

    UNINITIALIZED( tensorflow::ClientSession session    );
    UNINITIALIZED( tensorflow::EventsWriter event_writer );
};

#undef UNINITIALIZED

Network_state* network_init() {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    auto state = new Network_state;
    Scope& root = state->root;

    auto shape_scalar = Const<int>(root, 0, {0});

    // Network layout
    auto name = [&root](std::string str) { return root.WithOpName(str); };
    
    auto x    = Placeholder(name("x"),    DT_FLOAT, Placeholder::Shape({32, 256}));
    auto y    = Placeholder(name("y"),    DT_FLOAT, Placeholder::Shape({32}));
    auto rate = Placeholder(name("rate"), DT_FLOAT, Placeholder::Shape({}));

    auto w1 = Variable(name("w1"), {256, 64}, DT_FLOAT);
    auto b1 = Variable(name("b1"), {64}, DT_FLOAT);
    
    auto w2 = Variable(name("w2"), {64, 1}, DT_FLOAT);
    auto b2 = Variable(name("b2"), {1}, DT_FLOAT);

    auto a     = Tanh(name("a"),     BiasAdd(root, MatMul(root, x, w1), b1));
    auto y_out = Tanh(name("y_out"), BiasAdd(root, MatMul(root, a, w2), b2));

    auto loss = L2Loss(name("loss"), Sub(root, Reshape(root, y_out, {32}), y));

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
    add_init_op( Assign(init, w1, ParameterizedTruncatedNormal(init, {256, 64}, 0.f, .05f, -1.f, 1.f)) );
    add_init_op( Assign(init, w2, ParameterizedTruncatedNormal(init, { 64,  1}, 0.f, .05f, -1.f, 1.f)) );
    add_init_op( Assign(init, b1, Const<float>(init, 0.f, {64})) );
    add_init_op( Assign(init, b2, Const<float>(init, 0.f, {1}  )) );
    auto init_op = NoOp(init.WithControlDependencies(init_ops).WithOpName("run"));

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

struct Batch_data {
    Array_view<float> edge_weights;
    Array_view<float> results;
};

void network_batch(Network_state* state, Batch_data const& data) {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    assert(state);
    assert(data.edge_weights.size() == 32*256 and data.results.size() == 32);

    ++state->step;
    
    Tensor data_x {DT_FLOAT, {32, 256}};
    std::memcpy(data_x.flat<float>().data(), data.edge_weights.data(), data.edge_weights.size() * sizeof(float));
    
    Tensor data_y {DT_FLOAT, {32}};
    std::memcpy(data_y.flat<float>().data(), data.results.data(), data.results.size() * sizeof(float));
    
    std::vector<Tensor> outputs;
    TF_CHECK_OK(state->session.Run(
        {{state->x, data_x}, {state->y, data_y}, {state->rate, state->step < 200 ? 0.001f : 0.0001f}},
        {state->loss, state->summary_op},
        {state->update_op},
        &outputs
    ));
    
    std::cout << "Loss:  " << outputs[0].scalar<float>()() << endl;

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
    
    
    Event event;
    event.set_wall_time(elapsed_time());
    event.set_step(state->step);
    event.mutable_summary()->ParseFromString(outputs[1].scalar<std::string>()());
    state->event_writer.WriteEvent(event);
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
                nodes.push_back({i.other, i.weight});
            }
        }

        if (not nodes) {
            break;
        }

        auto f = [](u32 x) { return (x-1)*(x-1); };

        //jdbg >= nodes ,0;
        int sum = 0;
        for (int i = 0; i < nodes.size(); ++i) {
            sum += f(nodes[i].weight);
        }
        int arg = -1;
        if (sum == 0) {
            arg = jup_rand_uni(nodes.size());
        } else {        
            int val = jup_rand_uni(sum);
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


void network_main() {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    auto state_net = network_init();
    Graph_reader_state state_graph;
    graph_reader_init(&state_graph, "graph_data/small.schaf.lz4");

    constexpr int batch_size = 32;

    Neighbourhood_finder neighbours;
    
    Array<float> buffer;
    buffer.reserve(batch_size * (256 + 1));

    Array_view_mut<float> out_edge_weights {buffer.data(), batch_size * 256};
    Array_view_mut<float> out_results {out_edge_weights.end(), batch_size};
    
    int graph_left = 0;
    int cur_batch = 0;

    while (state_net->step < 500) {
        if (graph_left == 0) {
            bool c = graph_reader_next(&state_graph);
            if (not c) return;
            graph_left = state_graph.graph->num_nodes() / 32;
            jdbg < "Loading graph" < state_graph.graph->name.begin() < graph_left ,0;
            if (graph_left == 0) continue;
        }

        while (cur_batch < batch_size) {
            if (cur_batch == 0) {
                std::memset(out_edge_weights.data(), 0, out_edge_weights.size() * sizeof(float));
            }
            
            u32 node = jup_rand_uni(state_graph.graph->num_nodes());
            auto nodes = neighbours.find(*state_graph.graph, node, 16);

            /*
            for (int i = 0; i < nodes.size(); ++i) {
                for (int j = 0; j < nodes.size(); ++j) {
                    if (j == i) continue;
                    for (Edge e: state_graph.graph->adjacent(nodes[i])) {
                        if (e.other == nodes[j]) {
                            out_edge_weights[cur_batch*256 + i*16 + j] = e.weight;
                            break;
                        }
                    }
                }
                }*/
            
            for (int i = 0; i < nodes.size(); ++i) {
                int j = 0;
                if (nodes[i] == (u32)-1) continue;
                for (Edge e: state_graph.graph->adjacent(nodes[i])) {
                    while (j < nodes.size() and e.other > nodes[j]) ++j;
                    if (j >= nodes.size()) break;
                    if (e.other != nodes[j]) continue;
                    out_edge_weights[cur_batch*256 + i*16 + j] = e.weight;
                }
            }

            // Whether to have a positive or a negative example
            if (jup_rand_bool()) {
                /*
                for (int i = 0; i < 32; ++i) {
                    // Swap edges a and b
                    u32 a = jup_rand_uni(256 - 16);
                    u32 b = jup_rand_uni(256 - 16 - 1);
                    b += b >= a;
                    a += a / 16 + 1;
                    b += b / 16 + 1;

                    int a_ = (a % 16) * 16 + a / 16;
                    int b_ = (b % 16) * 16 + b / 16;

                    assert(out_edge_weights[cur_batch*256 + a] == out_edge_weights[cur_batch*256 + a_]);
                    assert(out_edge_weights[cur_batch*256 + b] == out_edge_weights[cur_batch*256 + b_]);
                    std::swap(out_edge_weights[cur_batch*256 + a ], out_edge_weights[cur_batch*256 + b ]);
                    std::swap(out_edge_weights[cur_batch*256 + a_], out_edge_weights[cur_batch*256 + b_]);
                }
                */
                
                for (int i = 0; i < 32; ++i) {
                    // Add 10 to a random edge
                    u32 a = jup_rand_uni(256 - 16);
                    a += a / 16 + 1;

                    out_edge_weights[cur_batch*256 + a] += 10;
                }

                out_results[cur_batch] = -1.f;
            } else {
                out_results[cur_batch] = 1.f;
            }

            // Normalize
            float max = 1.f;
            for (int i = 0; i < 256; ++i) {
                max = std::max(max, out_edge_weights[cur_batch*256 + i]);
            }
            //jdbg < max ,0;
            for (float& f: out_edge_weights.subview(cur_batch*256, 256)) {
                f /= max;
                //f = std::log(f/max*M_E + exp(-1));
            }
            /*for (int i = 0; i < 16; ++i) {
                jdbg >= out_edge_weights.subview(cur_batch*256 + i*16, 16) ,0;
            }
            jdbg ,0;
            if (cur_batch == 5) std::exit(0);*/
            
            
            ++cur_batch;
            --graph_left;

            if (graph_left == 0) break;
        }

        
        if (cur_batch == batch_size) {
            network_batch(state_net, Batch_data {out_edge_weights, out_results});
            
            cur_batch = 0;
        }
    }

    graph_reader_close(&state_graph);
    network_free(state_net);
}


void network_gendata(jup_str graph_file, Training_data* data) {
    assert(data);

    Graph_reader_state state_graph;
    graph_reader_init(&state_graph, graph_file);

    Neighbourhood_finder neighbours;
    
    int graph_left = 0;
    int cur_batch = 0;
    int cur_instance = 0;

    while (cur_batch < batch_count) {
        Batch_data& out = data->batches[cur_batch];
        
        if (graph_left == 0) {
            bool c = graph_reader_next(&state_graph);
            if (not c) {
                jout << "  Rewind for more data" << endl;
                graph_reader_reset(&state_graph);
                c = graph_reader_next(&state_graph);
                assert(c);
            }
                
            graph_left = state_graph.graph->num_nodes() / gen_graph_nodes;
            jdbg < "Loading graph" < state_graph.graph->name.begin() < graph_left ,0;
            if (graph_left == 0) continue;
        }

        while (cur_instance < batch_size) {
            if (cur_instance == 0) {
                std::memset(out.edge_weights.data(), 0, out.edge_weights.size() * sizeof(float));
            }
            
            u32 node = jup_rand_uni(state_graph.graph->num_nodes());
            auto nodes = neighbours.find(*state_graph.graph, node, 16);

            for (int i = 0; i < nodes.size(); ++i) {
                int j = 0;
                if (nodes[i] == (u32)-1) continue;
                for (Edge e: state_graph.graph->adjacent(nodes[i])) {
                    while (j < nodes.size() and e.other > nodes[j]) ++j;
                    if (j >= nodes.size()) break;
                    if (e.other != nodes[j]) continue;
                    out.edge_weights[cur_instance*256 + i*16 + j] = e.weight;
                }
            }

            // Whether to have a positive or a negative example
            if (jup_rand_bool()) {
                /*
                for (int i = 0; i < 32; ++i) {
                    // Swap edges a and b
                    u32 a = jup_rand_uni(256 - 16);
                    u32 b = jup_rand_uni(256 - 16 - 1);
                    b += b >= a;
                    a += a / 16 + 1;
                    b += b / 16 + 1;

                    int a_ = (a % 16) * 16 + a / 16;
                    int b_ = (b % 16) * 16 + b / 16;

                    assert(out.edge_weights[cur_instance*256 + a] == out.edge_weights[cur_instance*256 + a_]);
                    assert(out.edge_weights[cur_instance*256 + b] == out.edge_weights[cur_instance*256 + b_]);
                    std::swap(out.edge_weights[cur_instance*256 + a ], out.edge_weights[cur_instance*256 + b ]);
                    std::swap(out.edge_weights[cur_instance*256 + a_], out.edge_weights[cur_instance*256 + b_]);
                }
                */
                
                for (int i = 0; i < 32; ++i) {
                    // Add 10 to a random edge
                    u32 a = jup_rand_uni(256 - 16);
                    a += a / 16 + 1;

                    out.edge_weights[cur_instance*256 + a] += 10;
                }

                out.results[cur_instance] = -1.f;
            } else {
                out.results[cur_instance] = 1.f;
            }

            // Normalize
            float max = 1.f;
            for (int i = 0; i < 256; ++i) {
                max = std::max(max, out.edge_weights[cur_instance*256 + i]);
            }
            //jdbg < max ,0;
            for (float& f: out.edge_weights.subview(cur_instance*256, 256)) {
                f /= max;
                //f = std::log(f/max*M_E + exp(-1));
            }
            /*for (int i = 0; i < 16; ++i) {
                jdbg >= out.edge_weights.subview(cur_instance*256 + i*16, 16) ,0;
            }
            jdbg ,0;
            if (cur_instance == 5) std::exit(0);*/
            
            
            ++cur_instance;
            --graph_left;

            if (graph_left == 0) break;
        }

        
        if (cur_instance == batch_size) {
            cur_instance = 0;
            ++cur_batch;
        }
    }

    graph_reader_close(&state_graph);
    network_free(state_net);
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
