
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
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/events_writer.h"

#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

namespace tensorflow {
namespace ops {
namespace {

Status AddGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
    grad_outputs->push_back(Identity(scope, grad_inputs[0]));
    grad_outputs->push_back(Identity(scope, grad_inputs[0]));
    return scope.status();
}
REGISTER_GRADIENT_OP("Add", AddGrad);

Status SubGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
    grad_outputs->push_back(Identity(scope, grad_inputs[0]));
    grad_outputs->push_back(Neg(scope, grad_inputs[0]));
    return scope.status();
}
REGISTER_GRADIENT_OP("Sub", SubGrad);

Status L2LossGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
    grad_outputs->push_back(Mul(scope, op.input(0), grad_inputs[0]));
    return scope.status();
}
REGISTER_GRADIENT_OP("L2Loss", L2LossGrad);

Status BiasAddGrad_(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
    grad_outputs->push_back(grad_inputs[0]);
    grad_outputs->push_back(BiasAddGrad(scope, grad_inputs[0]));
    return scope.status();
}
REGISTER_GRADIENT_OP("BiasAdd", BiasAddGrad_);

}
} /* end of namespace ops */
} /* end of namespace tensorflow */

#pragma GCC diagnostic pop

namespace jup {

#define UNINITIALIZED(x) union {char JUP_UNIQUE_NAME(__dummy) = 0; x;}

struct Network_state {
    Network_state(): root {tensorflow::Scope::NewRootScope()} {}
    ~Network_state() {
        session.~ClientSession();
        event_writer.~EventsWriter();
    }
    
    tensorflow::Scope root;
    
    tensorflow::Operation update_op;
    tensorflow::Output x, y, rate, loss, summary;

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
    auto b2 = Variable(name("b2"), {}, DT_FLOAT);

    auto a     = Tanh(name("a"),     BiasAdd(root, MatMul(root, x, w1), b1));
    auto y_out = Tanh(name("y_out"), BiasAdd(root, MatMul(root, a, w2), b2));

    auto loss = L2Loss(name("loss"), Sub(root, Reshape(root, y_out, {32}), y));

    // Summaries
    Output s_loss = ScalarSummary(root, std::string {"s_loss"}, loss);
    auto summary = MergeSummary(name("summary"), {s_loss});

    // Update
    std::vector<Output> grad_vars {w1, b1, w2, b2};
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(root.NewSubScope("grad"), {loss}, grad_vars, &grad_outputs));
    
    auto update = root.NewSubScope("update");
    std::vector<Operation> update_ops;
    for (size_t i = 0; i < grad_vars.size(); ++i) {
        Output output = ApplyGradientDescent(update, grad_vars[i], rate, grad_outputs[i]);
        update_ops.push_back(output.op());
    }
    state->update_op = NoOp(update.WithControlDependencies(update_ops).WithOpName("run"));
    
    // Initialize
    auto init = root.NewSubScope("init");
    std::vector<Operation> init_ops;
    auto add_init_op = [&init_ops](Output output) { init_ops.push_back(output.op()); };
    add_init_op( Assign(init, w1, ParameterizedTruncatedNormal(init, {256, 64}, 0.f, .1f, -1.f, 1.f)) );
    add_init_op( Assign(init, w2, ParameterizedTruncatedNormal(init, { 64,  1}, 0.f, .1f, -1.f, 1.f)) );
    add_init_op( Assign(init, b1, Const<float>(init, 0.f, {64})) );
    add_init_op( Assign(init, b2, Const<float>(init, 0.f, {}  )) );
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
    state->x       = x;
    state->y       = y;
    state->rate    = rate;
    state->loss    = loss;
    state->summary = summary;

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

    ++state->step;

    assert(data.edge_weights.size() == 32*256 and data.results.size() == 32);
    
    Tensor data_x {DT_FLOAT, {32, 256}};
    std::memcpy(data_x.flat<float>().data(), data.edge_weights.data(), data.edge_weights.size());
    
    Tensor data_y {DT_FLOAT, {32}};
    std::memcpy(data_y.flat<float>().data(), data.results.data(), data.results.size());

    std::vector<Tensor> outputs;
    TF_CHECK_OK( state->session.Run(
        {{state->x, data_x}, {state->y, data_y}, {state->rate, 0.1f}},
        {state->loss, state->summary},
        {state->update_op},
        &outputs
    ) );
    
    std::cout << "Loss:  " << outputs[0].scalar<float>()() << '\n';
    
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

    state->event_writer.~EventsWriter();
    state->session.~ClientSession();
    delete state;
}

struct Neighbourhood_finder {
    struct Node_prio {
        u32 node;
        u32 weight;
    };

    Array<u32> result;
    Array<Node_prio> nodes;

    Array_view<u32> find(Graph const& graph, u32 node, int count);
};

Array_view<u32> Neighbourhood_finder::find(Graph const& graph, u32 node, int count) {
    nodes.reset();
    result.reset();

    result.push_back(node);

    while (result.size() < count) {
        u32 last = result.back();
        for (Edge i: graph.adjacent(last)) {
            if (result.count(i.other)) continue;

            bool found = false;
            for (auto& j: nodes) {
                if (j.node == i.other) {
                    j.weight += i.weight;
                    found = true;
                    break;
                }
            }
            if (not found) {
                nodes.push_back({i.other, i.weight});
            }
        }

        if (not nodes) {
            return result;
        }

        int max_arg = 0;
        for (int i = 0; i < nodes.size(); ++i) {
            if (nodes[i].weight > nodes[max_arg].weight) {
                max_arg = i;
            }
        }

        result.push_back(nodes[max_arg].node);
        nodes[max_arg] = nodes.back();
        nodes.addsize(-1);
    }


    while (result.size() < count) {
        result.push_back(graph.num_nodes());
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

    while (state_net->step < 5) {
        if (graph_left == 0) {
            bool c = graph_reader_next(&state_graph);
            if (not c) return;
            graph_left = state_graph.graph->num_nodes() / 32;
            if (graph_left == 0) continue;
        }

        while (cur_batch < batch_size) {
            u32 node = jup_rand_uni(state_graph.graph->num_nodes());
            auto nodes = neighbours.find(*state_graph.graph, node, 16);

            for (int i = 0; i < nodes.size(); ++i) {
                int j = 0;                
                for (Edge e: state_graph.graph->adjacent(nodes[i])) {
                    if (e.other != nodes[j]) continue;
                    out_edge_weights[cur_batch*256 + i*16 + j] = e.weight;
                    if (++j >= nodes.size()) break;
                }
                assert(j == nodes.size());
            }

            // Whether to have a positive or a negative example
            if (jup_rand_bool()) {
                for (int i = 0; i < 32; ++i) {
                    // Swap edges a and b
                    u32 a = jup_rand_uni(256);
                    u32 b = jup_rand_uni(256 - 1);
                    b += b >= a;

                    int a_ = (a % 16) * 16 + a / 16;
                    int b_ = (b % 16) * 16 + b / 16;

                    assert(out_edge_weights[cur_batch*256 + a] == out_edge_weights[cur_batch*256 + a_]);
                    assert(out_edge_weights[cur_batch*256 + b] == out_edge_weights[cur_batch*256 + b_]);
                    std::swap(out_edge_weights[cur_batch*256 + a ], out_edge_weights[cur_batch*256 + b ]);
                    std::swap(out_edge_weights[cur_batch*256 + a_], out_edge_weights[cur_batch*256 + b_]);
                }

                out_results[cur_batch] = -1.f;
            } else {
                out_results[cur_batch] = 1.f;
            }
            
            ++cur_batch;
            --graph_left;

            if (graph_left == 0) break;
        }

        
        if (cur_batch == batch_size) {
            network_batch(state_net, Batch_data {out_results, out_edge_weights});
            
            cur_batch = 0;
        }
    }

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
