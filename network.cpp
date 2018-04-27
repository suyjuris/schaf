
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

Status SoftmaxCrossEntropyWithLogitsGrad(const Scope& scope, const Operation& op, const std::vector<Output>& grad_inputs,
        std::vector<Output>* grad_outputs) {
    auto softmax_grad = op.output(1);
    grad_outputs->push_back(Mul(scope, softmax_grad, Stack(scope, {grad_inputs[0]}, Stack::Axis(-1))));
    grad_outputs->push_back(NoGradient());
    return scope.status();
}
REGISTER_GRADIENT_OP("SoftmaxCrossEntropyWithLogits", SoftmaxCrossEntropyWithLogitsGrad);

}
} /* end of namespace ops */
} /* end of namespace tensorflow */


namespace jup {

static void print_hyperparam(Hyperparam hyp) {
    jout << "Hyperparameters:\n"
         << "  batch_count:         " << hyp.batch_count << "\n"
         << "  batch_size:          " << hyp.batch_size << "\n"
         << "  recf_nodes:          " << hyp.recf_nodes << "\n"
         << "  recf_count:          " << hyp.recf_count << "\n"
         << "  gen_instances:       " << hyp.gen_instances << "\n"
         << "  learning_rate:       " << hyp.learning_rate << '\n'
         << "  learning_rate_decay: " << hyp.learning_rate_decay << '\n'
         << "  test_frac:           " << hyp.test_frac << '\n'
         << "  a1_size:             " << hyp.a1_size << '\n'
         << "  a2_size:             " << hyp.a2_size << '\n'
         << "  b1_size:             " << hyp.b1_size << '\n'
         << "  b2_size:             " << hyp.b2_size << '\n'
         << "  b3_size:             " << hyp.b3_size << '\n'
         << "  dropout:             " << hyp.dropout << '\n'
         << "  l2_reg:              " << hyp.l2_reg << '\n'
         << "  seed:                " << hyp.seed << '\n';
}

bool Hyperparam::valid() const {
    return  0   <  batch_count          and batch_count         <  1000000000
        and 0   <  batch_size           and batch_size          <  1000000
        and 0   <  recf_nodes           and recf_nodes          <  100
        and 0   <  recf_count           and recf_count          <  1000
        and 0   <  gen_instances        and gen_instances       <  1000000000
        and 0.f <= learning_rate        and learning_rate       <  1.f
        and std::isfinite(learning_rate)
        and 0   <= learning_rate_decay  and learning_rate_decay <  1000000
        and 0   <= test_frac            and test_frac           <= 1.f
        and std::isfinite(test_frac)
        and 0   <  a1_size              and a1_size             <  1000000
        and 0   <  a2_size              and a2_size             <  1000000
        and 0   <  b1_size              and b1_size             <  1000000
        and 0   <  b2_size              and b2_size             <  1000000
        and 0   <  dropout              and dropout             <= 1.f
        and std::isfinite(dropout)
        and 0   <= l2_reg               and l2_reg              <  100.f
        and std::isfinite(l2_reg)
        and _alignment >= 0 and _alignment % sizeof(float) == 0;
}

Batch_data Training_data::batch(int index) {
    Batch_data result;
    result.edge_weights = {&batch_data[index * hyp.floats_batch_aligned()], narrow<int>(hyp.floats_edge_weights())};
    result.results = {result.edge_weights.end(), narrow<int>(hyp.floats_results())};
    assert(result.results.end() <= batch_data.end());
    assert(result.results.end() == batch_data.begin() + index * hyp.floats_batch_aligned() + hyp.floats_batch());
    return result;
}
Batch_data Training_data::instance(int index) {
    int i_b = index / hyp.batch_size;
    int i_i = index % hyp.batch_size;
    Batch_data b = batch(i_b);
    int n = hyp.edges_instance();
    return {b.edge_weights.subview(i_i * n, n), b.results.subview(i_i, 1)};
}

Unique_ptr_free<Training_data> Training_data::make_unique(Hyperparam hyp, int realign_min_batchsize) {
    assert(realign_min_batchsize >= 0);
    
    if (not hyp.valid()) {
        print_hyperparam(hyp);
        die("Invalid hyperparameters in Training_data::make_unique! Something has gone horribly wrong...");
    }

    u64 size = sizeof(Training_data);
    if (realign_min_batchsize) {
        s64 max_batch_count = hyp.num_instances() / realign_min_batchsize;
        size += hyp.floats_batch() * hyp.batch_count * sizeof(float) + hyp.alignment() * max_batch_count;
    } else {
        size += hyp.bytes_total();
    }
    
    void* ptr = aligned_alloc(hyp.alignment(), size);
    std::memset(ptr, 0, size);
    Unique_ptr_free<Training_data> result {(Training_data*)ptr};
    result->hyp = hyp;
    result->batch_data.m_size = (size - sizeof(Training_data)) / sizeof(float);
    return result;
}

struct Allocator_Zero_copy: public tensorflow::Allocator {
    void* ptr = nullptr;
    u64 size = 0;

    Allocator_Zero_copy() {}
    
    template <typename T>
    Allocator_Zero_copy(Array_view_mut<T> arr) {
        ptr = arr.begin();
        size = arr.size() * sizeof(T);
    }

    std::string Name() override { return "Allocator_Zero_Copy"; }
    
    void* AllocateRaw(size_t alignment, size_t num_bytes) override {
        assert((u64)ptr % alignment == 0);
        assert(num_bytes == size);
        if ((u64)ptr % alignment == 0 and num_bytes == size) {
            return ptr;
        }
        return nullptr;
    }

    void DeallocateRaw(void* ptr) override {}
};

struct Batch_tensors {
    tensorflow::Tensor data_x, data_y;
    Allocator_Zero_copy data_x_alloc, data_y_alloc;
};

#define UNINITIALIZED(x) union {char JUP_UNIQUE_NAME(__dummy) = 0; x;}

struct Network_state {
    Network_state(): root {tensorflow::Scope::NewRootScope()} {}
    ~Network_state() {
        if (event_writer_initialised) {
            event_writer.~EventsWriter();
        }
        session.~ClientSession();
    }

    Hyperparam hyp;
    
    tensorflow::Scope root;
    
    tensorflow::Operation update_op, save_op, restore_op, decay_op, dropout_on_op, dropout_off_op,
        print_op;
    tensorflow::Output x, y, y_prob, rate, summary_op, loss_train, loss_train_avg, loss_test, loss_comp, f1_score;
    
    std::vector<tensorflow::Output> params;

    int step = 0;
    int epoch = 0;
    int epoch_start = 0;

    Timer timer;
    float loss_sum1 = 0.f;
    float loss_sum2 = 0.f;
    int loss_count1 = 0;
    int loss_count2 = 0;
    Batch_tensors last_batch;

    std::string save_path;

    UNINITIALIZED( tensorflow::ClientSession session     );
    UNINITIALIZED( tensorflow::EventsWriter event_writer );
    bool event_writer_initialised = false;

    u64 last_id = 0;
    int next_seed() {
        if (not hyp.seed) return 0;
        last_id += hyp.seed + last_id == 0;
        return hyp.seed + last_id++;
    }
};

#undef UNINITIALIZED

static tensorflow::Output Dropout(Network_state* state, tensorflow::Input x, tensorflow::Input p) {
    assert(state);
    
    using namespace tensorflow;
    using namespace tensorflow::ops;
    
    // Similar to the python dropout, see tensorflow/python/ops/nn_ops.py
    auto dropout = state->root.NewSubScope("dropout");
    auto binary = Floor(dropout, Add(dropout, RandomUniform(dropout, Shape(dropout, x), DT_FLOAT,
        RandomUniform::Seed(state->next_seed())), p));
    return Mul(dropout, Div(dropout, x, p), binary);
}

static tensorflow::Output BinarySoftmaxCrossEntropy(tensorflow::Scope scope, tensorflow::Input logits, tensorflow::Input label) {
    using namespace tensorflow;
    using namespace tensorflow::ops;
    
    Input bin_pos = Mul(scope, Add(scope, label,  1.f),  0.5f);
    Input bin_neg = Mul(scope, Add(scope, label, -1.f), -0.5f);
    Input logits_ = Mul(scope, logits, -1.f);
    auto softmax = SoftmaxCrossEntropyWithLogits(scope, Stack(scope, {logits, logits_}, Stack::Axis(-1)), Stack(scope, {bin_pos, bin_neg}, Stack::Axis(-1)));
    return Sum(scope, softmax.loss, {0});
}

Network_state* network_init(Hyperparam hyp, bool disable_saving) {
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

    auto a1tmp = Reshape(root, x, {hyp.batch_size * hyp.recf_count, narrow<int>(hyp.edges_recf())});
    auto a1out = Dropout(state, Tanh(name("a1out"), BiasAdd(root, MatMul(root, a1tmp, a1w), a1b)), drop);
    auto a2out = Dropout(state, Tanh(name("a2out"), BiasAdd(root, MatMul(root, a1out, a2w), a2b)), drop);

    auto b1tmp = Reshape(root, a2out, {hyp.batch_size, hyp.recf_count * hyp.a2_size});
    auto b1out = Dropout(state, Tanh(name("b1out"), BiasAdd(root, MatMul(root, b1tmp, b1w), b1b)), drop);
    auto b2out = Dropout(state, Tanh(name("b2out"), BiasAdd(root, MatMul(root, b1out, b2w), b2b)), drop);
    //auto y_out = Reshape(root, Tanh(name("y_out"), BiasAdd(root, MatMul(root, b2out, b3w), b3b)), {hyp.batch_size});
    //auto y_prob = y_out;
    auto y_out = Reshape(root, BiasAdd(name("y_out"), MatMul(root, b2out, b3w), b3b), {hyp.batch_size});
    auto y_prob = Add(name("y_prob"), Mul(root, Sigmoid(root, y_out), 2.f), -1.f);

    s64 num_parameters = hyp.edges_recf() * hyp.a1_size + hyp.a1_size + hyp.a1_size * hyp.a2_size
        + hyp.a2_size + hyp.a2_size * hyp.recf_count * hyp.b1_size + hyp.b1_size + hyp.b1_size * hyp.b2_size
        + hyp.b2_size + hyp.b2_size * hyp.b3_size + hyp.b3_size;
    auto loss = Add(name("loss"),
        //Mul(root, L2Loss(root, Sub(root, y_out, y)), batch_size_inv),
        Mul(root, BinarySoftmaxCrossEntropy(root, y_out, y), batch_size_inv),
        Mul(root, AddN(root, std::initializer_list<Output> {
            L2Loss(root, a1w), L2Loss(root, a2w), L2Loss(root, b1w), L2Loss(root, b2w), L2Loss(root, b3w)
        }), hyp.l2_reg / num_parameters)
    );

    // Update
    std::vector<Output> grad_vars {a1w, a1b, a2w, a2b, b1w, b1b, b2w, b2b, b3w, b3b};
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(root.NewSubScope("grad"), {loss}, grad_vars, &grad_outputs));
    
    auto update = root.NewSubScope("update");
    std::vector<Operation> update_ops;
    for (size_t i = 0; i < grad_vars.size(); ++i) {
        Output output = ApplyGradientDescent(update, grad_vars[i], rate, grad_outputs[i]);
        update_ops.push_back(output.op());
    }
    state->update_op = NoOp(update.WithControlDependencies(update_ops).WithOpName("run"));

    // Summaries
    auto summary = root.NewSubScope("summary");
    std::vector<Output> summary_ops;

    state->loss_train_avg = Placeholder(name("loss_train_avg"), DT_FLOAT, Placeholder::Shape({}));
    state->loss_test      = Placeholder(name("loss_test"),      DT_FLOAT, Placeholder::Shape({}));
    state->loss_comp      = Placeholder(name("loss_comp"),      DT_FLOAT, Placeholder::Shape({}));
    state->f1_score       = Placeholder(name("f1_score"),       DT_FLOAT, Placeholder::Shape({}));
    
    auto add_summary = [&summary_ops](Output out) { summary_ops.push_back(out); };
    add_summary( ScalarSummary(root, std::string {"s_rate"}, rate));
    add_summary( ScalarSummary(root, std::string {"s_loss_train"}, state->loss_train_avg));
    add_summary( ScalarSummary(root, std::string {"s_loss_test"},  state->loss_test     ));
    add_summary( ScalarSummary(root, std::string {"s_loss_comp"},  state->loss_comp     ));
    add_summary( ScalarSummary(root, std::string {"s_f1_score"},   state->f1_score      ));
    add_summary( HistogramSummary(root, std::string {"s_x"},    x)    );
    add_summary( HistogramSummary(root, std::string {"s_y"},    y)    );
    add_summary( HistogramSummary(root, std::string {"s_yout"}, y_out));
    add_summary( HistogramSummary(root, std::string {"s_a1w"}, a1w)   );
    add_summary( HistogramSummary(root, std::string {"s_a1b"}, a1b)   );
    add_summary( HistogramSummary(root, std::string {"s_a2w"}, a2w)   );
    add_summary( HistogramSummary(root, std::string {"s_a2b"}, a2b)   );
    add_summary( HistogramSummary(root, std::string {"s_b1w"}, b1w)   );
    add_summary( HistogramSummary(root, std::string {"s_b1b"}, b1b)   );
    add_summary( HistogramSummary(root, std::string {"s_b2w"}, b2w)   );
    add_summary( HistogramSummary(root, std::string {"s_b2b"}, b2b)   );
    add_summary( HistogramSummary(root, std::string {"s_b3w"}, b3w)   );
    add_summary( HistogramSummary(root, std::string {"s_b3b"}, b3b)   );
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
        auto o = ParameterizedTruncatedNormal::Seed(state->next_seed());
        add_init_op( Assign(init, i, ParameterizedTruncatedNormal(init, Shape(init, i), 0.f, .05f, -1.f, 1.f, o)) );
    }
    for (auto i: {a1b, a2b, b1b, b2b, b3b}) {
        add_init_op( Assign(init, i, Fill(init, Shape(init, i), 0.f)) );
    }
    add_init_op( Assign(init, rate, Const<float>(init, hyp.learning_rate, {})) );
    add_init_op( Assign(init, drop, Const<float>(init, hyp.dropout, {})) );
    auto init_op = NoOp(init.WithControlDependencies(init_ops).WithOpName("run"));

    // Dropout initialization
    auto dropout = root.NewSubScope("dropout_set");
    state->dropout_on_op  = Output { Assign(dropout, drop, Const<float>(dropout, hyp.dropout, {})) }.op();
    state->dropout_off_op = Output { Assign(dropout, drop, Const<float>(dropout, 1.f, {})) }.op();

    // Learning rate decay
    auto decay = root.NewSubScope("decay");
    state->decay_op = Output {Assign(decay, rate, Mul(decay, rate, 0.5f))}.op();
    
    // Checkpoints
    auto checkpoint = root.NewSubScope("checkpoint");
    std::vector<Output> params_op       { a1w,   a1b,   a2w,   a2b,   b1w,   b1b,   b2w,   b2b,   b3w,   b3b };
    std::vector<std::string> params_str {"a1w", "a1b", "a2w", "a2b", "b1w", "b1b", "b2w", "b2b", "b3w", "b3b"};

    Tensor params_name  {DT_STRING, {(int)params_op.size()}}; 
    Tensor params_slice {DT_STRING, {(int)params_op.size()}}; 
    std::vector<DataType> params_type {params_op.size(), DT_FLOAT};
    for (int i = 0; i < (int)params_op.size(); ++i) {
        params_name.flat<std::string>()(i) = params_str[i];
    }
    if (not disable_saving and global_options.logdir and global_options.iter_save) {
        std__filesystem::path path {global_options.logdir.c_str()};
        path /= get_date_string().c_str();
        std::error_code ec;
        std__filesystem::create_directories(path, ec);
        if (ec) {
            die("%s\nwhile trying to create directory %s", ec.message(), path.c_str());
        }

        state->save_path = (path / "param").native();
        state->save_op = SaveV2(checkpoint.WithOpName("save"), std::string {state->save_path}, params_name,
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

    if (not disable_saving and global_options.logdir and global_options.iter_event) {
        std__filesystem::path path {global_options.logdir.c_str()};
        path /= get_date_string().c_str();
        std::error_code ec;
        std__filesystem::create_directories(path, ec);
        if (ec) {
            die("%s\nwhile trying to create directory %s", ec.message(), path.c_str());
        }

        new (&state->event_writer) EventsWriter {(path / "run").native()};
        assert( state->event_writer.Init() );
        state->event_writer_initialised = true;
        state->event_writer.WriteEvent(event);
    }

    SessionOptions session_options;
    //session_options.config.set_log_device_placement(true);
    
    // Initialize session
    new (&state->session) ClientSession {root, session_options};

    state->x           = x;
    state->y           = y;
    state->y_prob      = y_prob;
    state->rate        = rate;
    state->loss_train  = loss;
    
    state->session.Run({}, {}, {init_op}, nullptr);

    return state;
}

static void hyp_init_alignment(Hyperparam* hyp) {
    hyp->_alignment = tensorflow::Allocator::kAllocatorAlignment;
    if (hyp->_alignment % sizeof(float) != 0) {
        // Could just set the alignment to a multiple that does...
        die("Somehow tensorflow alignment does not include float alignment.");
    }
}

static void fill_tensors(Hyperparam hyp, Batch_data const& batch, Batch_tensors* into) {
    using namespace tensorflow;

    into->data_x_alloc = {batch.edge_weights};
    into->data_y_alloc = {batch.results};
    
    into->data_x = Tensor {&into->data_x_alloc, DT_FLOAT, {hyp.batch_size, hyp.recf_count, hyp.edges_recf()}};
    into->data_y = Tensor {&into->data_y_alloc, DT_FLOAT, {hyp.batch_size}};

    // The following code does not need to have the memory aligned, but is really SLOW.
    //if (data_x->NumElements() == 0) {
    //    *data_x = Tensor {DT_FLOAT, {hyp.batch_size, hyp.recf_count, hyp.edges_recf()}};
    //}
    //if (data_y->NumElements() == 0) {
    //    *data_y = Tensor {DT_FLOAT, {hyp.batch_size}};
    //}
    //
    //assert(batch.edge_weights.size() == data_x->NumElements());
    //std::memcpy(data_x->flat<float>().data(), batch.edge_weights.data(), batch.edge_weights.size() * sizeof(float));
    //
    //assert(batch.results.size() == data_y->NumElements());
    //std::memcpy(data_y->flat<float>().data(), batch.results.data(), batch.results.size() * sizeof(float));    
}

static void network_batch(Network_state* state, Batch_data const& data, bool silent) {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    assert(state);

    auto hyp = state->hyp;

    Batch_tensors& tensors = state->last_batch;
    fill_tensors(hyp, data, &tensors);

    if (hyp.learning_rate_decay
        and state->epoch % hyp.learning_rate_decay == 0
        and state->epoch_start == state->step
        and state->step > 0
    ) {
        std::vector<Tensor> outputs;
        TF_CHECK_OK(state->session.Run({}, {}, {state->decay_op}, nullptr));
        TF_CHECK_OK(state->session.Run({}, {state->rate}, {}, &outputs));
        if (not silent) {
            jout << "  Reducing learning rate, down to " << outputs[0].scalar<float>()() << "\n";
        }
    }
    
    ++state->step;
    
    std::vector<Tensor> outputs;
    TF_CHECK_OK(state->session.Run(
        {{state->x, tensors.data_x}, {state->y, tensors.data_y}},
        {state->loss_train},
        {state->update_op},
        &outputs
    ));

    state->loss_sum1 += outputs[0].scalar<float>()();
    state->loss_count1++;

    state->loss_sum2 += outputs[0].scalar<float>()();
    state->loss_count2++;
}

struct Network_validate_result {
    float loss_test       = 0.f; // Average loss on the test dataset
    float loss_test_l2    = 0.f; // Average MSE on the test dataset
    
    float false_negative  = 0.f; // Percentage of wrongly rejected instances (for cutoff of 0.5)
    float false_positive  = 0.f; // Percentage of wrongly accepted instances (for cutoff of 0.5)
    float wrong_random    = 0.f; // Like false_positive, but only consider the random instances
    float wrong_perturbed = 0.f; // Like false_positive, but only consider the perturbed instances

    float total_negative    = 0.f;
    float total_positive    = 0.f;
    float total_random      = 0.f;
    float total_perturbed   = 0.f;

    float false_negative_frac()  const { return false_negative  / total_negative;  }
    float false_positive_frac()  const { return false_positive  / total_positive;  }
    float wrong_random_frac()    const { return wrong_random    / total_random;    }
    float wrong_perturbed_frac() const { return wrong_perturbed / total_perturbed; }

    float precision() const {
        return 1.f - false_positive / total_positive;
    }
    float recall() const {
        return 1.f - false_negative / (total_positive - false_positive + false_negative);
    }
    float f1_score() const {
        return 2 * precision() * recall() / (precision() + recall());
    }

    void print(std::ostream& out) {
        out << "  Loss (validation): " << jup_printf("%.4f\n", (double)loss_test)
            << "  Loss (valid.,l2):  " << jup_printf("%.4f\n", (double)loss_test_l2)
            << "  False negatives:   " << jup_printf("%.4f (%.0f/%.0f)\n", (double)false_negative_frac(),
                (double)false_negative,  (double)total_negative)
            << "  False positives:   " << jup_printf("%.4f (%.0f/%.0f)\n", (double)false_positive_frac(),
                (double)false_positive,  (double)total_positive)
            << "  Wrong positive:    " << jup_printf("%.4f (%.0f/%.0f)\n", (double)(1.f - recall()),
                (double)false_negative,    (double)(total_positive - false_positive + false_negative))
            << "  Wrong random:      " << jup_printf("%.4f (%.0f/%.0f)\n", (double)wrong_random_frac(),
                (double)wrong_random,    (double)total_random)
            << "  Wrong perturbed:   " << jup_printf("%.4f (%.0f/%.0f)\n", (double)wrong_perturbed_frac(),
                (double)wrong_perturbed, (double)total_perturbed)
            << "  Precision:         " << jup_printf("%.4f\n", (double)precision())
            << "  Recall:            " << jup_printf("%.4f\n", (double)recall())
            << "  F1-score:          " << jup_printf("%.4f\n", (double)f1_score());
    }
};

static Network_validate_result network_validate(Network_state* state, Training_data /*const*/& data_test) {
    using namespace tensorflow;
    
    Network_validate_result result;
    
    // Need to turn off dropout during validation
    TF_CHECK_OK(state->session.Run({}, {}, {state->dropout_off_op}, nullptr));

    Batch_tensors tensors;
    std::vector<Tensor> outputs;
    for (int i = 0; i < data_test.hyp.batch_count; ++i) {
        fill_tensors(data_test.hyp, data_test.batch(i), &tensors);
        TF_CHECK_OK(state->session.Run(
            {{state->x, tensors.data_x}, {state->y, tensors.data_y}},
            {state->loss_train, state->y_prob},
            {},
            &outputs
        ));
        
        auto res_real = data_test.batch(i).results;
        auto res_yout = outputs[1].vec<float>();
        float f = std::nextafter(-1.f, 0.f);

        result.loss_test += outputs[0].scalar<float>()();
        for (int i = 0; i < data_test.hyp.batch_size; ++i) {
            if (res_real[i] == 0) continue;
            assert(res_real[i] == 1.f or res_real[i] == -1.f or res_real[i] == f);
            result.loss_test_l2    += 0.5 * std::pow(res_real[i] - res_yout(i), 2.0);
            result.false_negative  += res_real[i] ==  1.f and res_yout(i) <= 0.f;
            result.false_positive  += res_real[i] <=    f and res_yout(i) >  0.f;
            result.wrong_random    += res_real[i] == -1.f and res_yout(i) >  0.f;
            result.wrong_perturbed += res_real[i] ==    f and res_yout(i) >  0.f;
            result.total_negative  += res_yout(i) <= 0.f;
            result.total_positive  += res_yout(i) >  0.f;
            result.total_random    += res_real[i] == -1.f;
            result.total_perturbed += res_real[i] ==    f;
        }
    }
    
    TF_CHECK_OK(state->session.Run({}, {}, {state->dropout_on_op}, nullptr));

    result.loss_test    /= data_test.hyp.batch_count;
    result.loss_test_l2 /= data_test.hyp.num_instances();
    
    return result;
}

static void network_summary(Network_state* state, Training_data /*const*/& data_test) {
    using namespace tensorflow;
    
    assert(state);

    auto vali = network_validate(state, data_test);

    float loss_train = state->loss_sum2 / state->loss_count2;
    state->loss_sum2 = 0.f;
    state->loss_count2 = 0;

    std::vector<Tensor> outputs;
    TF_CHECK_OK(state->session.Run(
        {{state->x, state->last_batch.data_x}, {state->y, state->last_batch.data_y},
         {state->loss_train_avg, loss_train}, {state->loss_test, vali.loss_test},
         {state->loss_comp, vali.loss_test_l2}, {state->f1_score, vali.f1_score()}},
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

static void fill_instance(
    Graph const& graph,
    Neighbourhood_finder* neighbours,
    Hyperparam hyp,
    Batch_data* out,
    int cur_instance,
    bool do_falsify
) {
    assert(neighbours and out);
    
    for (int cur_recf = 0; cur_recf < hyp.recf_count; ++cur_recf) {
        u32 node = global_rng.gen_uni(graph.num_nodes());
        auto nodes = do_falsify ?
            neighbours->find<true >(graph, node, hyp.recf_nodes):
            neighbours->find<false>(graph, node, hyp.recf_nodes);

        // Ignore empty slots at the end
        int nodes_end = nodes.size();
        while (nodes_end > 0 and nodes[nodes_end - 1] == (u32)-1) --nodes_end;

        // Fill in the edges
        for (int i = 0; i+1 < nodes_end; ++i) {
            for (Edge e: graph.adjacent(nodes[i])) {
                for (int j = i+1; j < nodes_end; ++j) {
                    if (e.other != nodes[j]) continue;
                    u32 weight = do_falsify ? perturb(nodes[i], nodes[j], e.weight) : e.weight;
                    out->edge(cur_instance, cur_recf, i, j, hyp) = weight;
                    out->edge(cur_instance, cur_recf, j, i, hyp) = weight;
                }
            }
        }
    }
}

void network_generate_data(jup_str graph_file, Training_data* data) {
    assert(data);

    if (not data->hyp.valid()) {
        print_hyperparam(data->hyp);
        die("Invalid hyperparameters in network_generate_data, something has gone horribly wrong!");
    }

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

            if (state_graph.graph->num_nodes() == 0) continue;
            
            graph_left = data->hyp.gen_instances;
        }

        if (timer.update()) {
            auto s = jup_printf(" (batch %d/%d, instance %d/%d, graph %s, ", cur_batch, data->hyp.batch_count,
                cur_instance, data->hyp.batch_size, state_graph.graph->name.begin());
            jout << "  Currently at " << timer.progress(cur_batch * data->hyp.batch_size + cur_instance) << s
                 << timer.bytes(cur_batch * data->hyp.bytes_batch_aligned() + cur_instance * data->hyp.bytes_instance())
                 << ")" << endl;
        }

        while (cur_instance < data->hyp.batch_size) {
            if (cur_instance == 0) {
                jup_memset(&out.edge_weights);
            }

            bool do_falsify = not cur_graph_random and global_rng.gen_bool(256 / 3);

            fill_instance(*state_graph.graph, &neighbours, data->hyp, &out, cur_instance, do_falsify);

            // Whether to have a positive or a negative example
            if (do_falsify) {
                // Hacky: Change the result a little bit to indicate whether the instance was random
                out.results[cur_instance] = std::nextafter(-1.f, 0.f);
            } else if (cur_graph_random) {
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

    jout << "Done. (" << timer.total() << ", " << timer.bytes_done(cur_batch * data->hyp.bytes_batch_aligned())
         << ", " << num_graph << " ordinary graphs, " << num_random << " random graphs, " << num_rewind
         << " rewinds)" << endl;
}

void network_shuffle(Training_data const* from, Training_data* into, int offset, bool silent) {
    assert(into);
    assert(from->hyp.valid() and into->hyp.valid());
    assert(0 <= offset and offset < from->hyp.num_instances());

    if (from->hyp.recf_nodes != into->hyp.recf_nodes) {
        die("Incompatible sets of hyperparameters, different number of nodes per receptive field "
            "(have: %d, want: %d)", from->hyp.recf_nodes, into->hyp.recf_nodes+0 /* packing */);
    } else if (from->hyp.recf_count != into->hyp.recf_count) {
        die("Incompatible sets of hyperparameters, different number receptive fields "
            "(have: %d, want: %d)", from->hyp.recf_count, into->hyp.recf_count+0 /* packing */);
    } else if (from->hyp.num_instances() - offset < into->hyp.num_instances()) {
        die("Incompatible sets of hyperparameters, there is not enough data to fill the target "
            "(have: %d instances, want: %d instances)", (int)from->hyp.num_instances() - offset,
            (int)into->hyp.num_instances());
    }
    
    Timer timer;

    Array<int> left;
    left.resize(into->hyp.num_instances());
    for (int i = 0; i < left.size(); ++i) left[i] = i;

    Rng rng {global_rng.rand()};
    for (int j = offset; j < offset + into->hyp.num_instances(); ++j) {
        int index = rng.gen_uni(left.size());
        int i = left[index];
        left[index] = left.back();
        left.pop_back();

        // Move instance j into instance i
        auto j_inst = const_cast<Training_data*>(from)->instance(j);
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
    assert(data_train);

    jout << "  Loading training data... (" << data_file << ")" << endl;

    // Load the file
    Hyperparam hyp_td;
    load_bytes(data_file, &hyp_td, -1);
    auto data_orig = Training_data::make_unique(hyp_td);
    load_bytes(data_file, data_orig.get(), Training_data::bytes_extra(hyp_td));

    hyp_init_alignment(&hyp);
    
    Hyperparam hyp_train = hyp;
    Hyperparam hyp_test  = hyp;
    hyp_test.batch_count = data_test ? hyp.test_frac * hyp.batch_count : 0;
    hyp_train.batch_count = hyp.batch_count - hyp_test.batch_count;
    
    // Check that there is enough data
    if (hyp_td.num_instances() < hyp_train.num_instances() + hyp_test.num_instances()) {
        die("Set of training data is too small, there is not enough data to fill the target "
        "(have: %d instances, want: %d instances)", (int)hyp_td.num_instances(),
        (int)(hyp_train.num_instances() + hyp_test.num_instances()));
    }

    if (data_test) {
        jout << "  Using " << hyp_train.batch_count << " batches for training, "
             << hyp_test.batch_count << " for testing\n";
    } else {
        jout << "  Using " << hyp_train.batch_count << " batches in total\n";
    }
    
    // Do the copying via shuffle (which also checks that the sizes are correct and handles
    // different values of batch_size correctly

    *data_train = Training_data::make_unique(hyp_train);
    network_shuffle(data_orig.get(), data_train->get(), 0, true);
    
    if (data_test) {
        *data_test  = Training_data::make_unique(hyp_test );
        network_shuffle(data_orig.get(), data_test->get(), (**data_train).hyp.num_instances(), true);
    }
    
    u64 hash = hash_shuffle_inv(*data_orig, hyp.num_instances());
    if (data_test) {
        assert(hash == hash_shuffle_inv(**data_train) + hash_shuffle_inv(**data_test));
    } else {
        assert(hash == hash_shuffle_inv(**data_train));
    }
}

struct Async_shuffler {
    Training_data const* from;
    Training_data* into;
    int offset;
    bool received_params;
    bool keep_alive;
    
    std::thread thread;
    std::mutex shuffling;
    std::condition_variable cv_swap, cv_start;
};

void async_shuffle_init(Async_shuffler* shuffler) {
    assert(not shuffler->thread.joinable());
    
    auto func = [shuffler] () {
        std::unique_lock<std::mutex> lock {shuffler->shuffling};
        shuffler->cv_start.notify_all();
        while (true) {
            while (shuffler->keep_alive and not shuffler->received_params) {
                shuffler->cv_swap.wait(lock);
            }
            if (not shuffler->keep_alive) break;
            shuffler->received_params = false;
            network_shuffle(shuffler->from, shuffler->into, shuffler->offset, true);
        }
    };

    std::unique_lock<std::mutex> lock {shuffler->shuffling};
    
    shuffler->keep_alive = true;
    shuffler->received_params = false;
    shuffler->thread = std::thread {func};
    shuffler->cv_start.wait(lock);
}

void async_shuffle_start(Async_shuffler* shuffler, Training_data const* from, Training_data* into, int offset) {
    assert(shuffler->thread.joinable());
    
    shuffler->from = from;
    shuffler->into = into;
    shuffler->offset = offset;
    shuffler->received_params = true;

    shuffler->shuffling.unlock();
    shuffler->cv_swap.notify_all();
}

void async_shuffle_wait(Async_shuffler* shuffler) {
    assert(shuffler->thread.joinable());
    
    shuffler->shuffling.lock();
}

void async_shuffle_close(Async_shuffler* shuffler) {
    assert(shuffler->thread.joinable());
    
    async_shuffle_wait(shuffler);
    shuffler->keep_alive = false;
    shuffler->shuffling.unlock();
    shuffler->cv_swap.notify_all();
    
    shuffler->thread.join();
}

void network_train(jup_str data_file) {
    auto hyp = global_options.hyp;
    print_hyperparam(hyp);

    jout << "Initializing..." << endl;
    // Shut tensorflow up
    setenv("TF_CPP_MIN_LOG_LEVEL", "1", 0);
    init_signal_sigint();

    jout << "  Creating neural network..." << endl;
    auto state = network_init(hyp);

    Unique_ptr_free<Training_data> data_train, data_test, data_tmp;
    network_load_data(data_file, hyp, &data_train, &data_test);
    data_tmp = Training_data::make_unique(data_train->hyp);

    if (global_options.param_in) {
        jout << "  Restoring from checkpoint... (" << global_options.param_in << ")" << endl;
        network_restore(state);
    }

    Async_shuffler shuffler;
    async_shuffle_init(&shuffler);
    async_shuffle_start(&shuffler, data_train.get(), data_tmp.get(), 0);

    jout << "Training..." << endl;
    //int initial_step = state->step;
    int cur_batch = 0;
    while (state->step < global_options.iter_max) {
        network_batch(state, data_train->batch(cur_batch), false);
        ++cur_batch;
        
        if (state->step
            and global_options.iter_save
            and global_options.logdir
            and state->step % global_options.iter_save == 0
        ) {
            network_save(state);
        }

        if (state->step
            and global_options.iter_event
            and state->step % global_options.iter_event == 0
        ) {
            network_summary(state, *data_test);
        }

        if (cur_batch >= data_train->hyp.batch_count) {
            async_shuffle_wait(&shuffler);
            std::swap(data_train, data_tmp);
            async_shuffle_start(&shuffler, data_train.get(), data_tmp.get(), 0);
            cur_batch = 0;
            ++state->epoch;
            state->epoch_start = state->step;
        }
        
        if (state->timer.update()) {
            auto vali = network_validate(state, *data_test);
            jout << jup_printf(
                "  Loss: %.4f train, %.4f test, %.4f comp, (errors pos/rand/per: %.2f/%.2f/%.2f",
                (double)(state->loss_sum1 / state->loss_count1), (double)vali.loss_test,
                (double)vali.loss_test_l2, (double)(1.f - vali.recall()),
                (double)vali.wrong_random_frac(), (double)vali.wrong_perturbed_frac()
            ) << ", epoch " << state->epoch << ", iteration " << state->step;
            if (global_options.iter_max < std::numeric_limits<int>::max()) {
                jout << "/" << global_options.iter_max;
            }
            jout << ", " << state->timer.counter(state->step) << " iter/s)" << endl;

            state->loss_sum1 = 0;
            state->loss_count1 = 0;
        }

        if (global_interrupt_flag) break;
    }

    async_shuffle_close(&shuffler);
    
    if (global_options.logdir) {
        jout << "Saving parameters to \"" << state->save_path << "\"... \n";
        network_save(state);
    }
    
    auto vali = network_validate(state, *data_test);
    jout << "Final results:\n"
         << "  Loss (training):   " << jup_printf("%.4f\n", state->loss_sum1 / state->loss_count1);
    vali.print(jout);
    jout << "  Total time:        " << state->timer.total() << '\n'
         << "  Total iterations:  " << state->step << '\n'
         << "  Average speed:     " << state->timer.counter_done(state->step) << " iter/s" << endl;

    network_free(state);
}

void network_print_data_info(jup_str data_file) {
    jout << "Loading file " << data_file << "..." << endl;
    
    Hyperparam hyp;
    load_bytes(data_file, &hyp, -1);

    u64 size = get_file_size(data_file);
    s64 size_td = Training_data::bytes_total(hyp);
    if ((u64)size_td != size) {
        die("Invalid file size. Expected %d bytes, got %" PRId64 ".", size_td, size);
    }
    
    auto data = Training_data::make_unique(hyp);
    load_bytes(data_file, data.get(), Training_data::bytes_extra(hyp));

    u64 hash_val_oi = hash_shuffle_inv(*data);
    u64 hash_val = XXH64(data.get(), Training_data::bytes_total(data->hyp), 0);

    jout << "\n";
    print_hyperparam(hyp);

    int count_positive  = 0;
    int count_random    = 0;
    int count_perturbed = 0;
    int count_error     = 0;

    for (int i = 0; i < hyp.num_instances(); ++i) {
        float f = std::nextafter(-1.f, 0.f);
        auto instance = data->instance(i);

        count_positive  += instance.results[0] ==  1.f;
        count_random    += instance.results[0] == -1.f;
        count_perturbed += instance.results[0] ==    f;
        count_error     += (instance.results[0] != f and instance.results[0] != -1.f and instance.results[0] != 1.f);
    }

    jout << "\nStatistics:\n";
    jout << "  total floats:        " << hyp.floats_total() << "\n"
         << "  total bytes:         " << nice_bytes(size_td) << "\n"
         << "  instances positive:  " << count_positive << "\n"
         << "  instances negative:  " << count_random + count_perturbed << "\n"
         << "  instances random:    " << count_random << "\n"
         << "  instances perturbed: " << count_perturbed << "\n"
         << "  instances invalid:   " << count_error << "\n"
         << "  checksum (xxHash64): " << nice_hex(hash_val) << '\n'
         << "  checksum (s.i.):     " << nice_hex(hash_val_oi) << '\n';
}

void network_random_hyp(Hyperparam* hyp, Rng* rng) {
    int inst_total = hyp->num_instances();
    hyp->batch_size = rng->gen_uni(64, 512 + 1);
    hyp->batch_count = inst_total / hyp->batch_size;
    
    hyp->learning_rate = rng->gen_uni_float() * 0.32 + 0.03;
    hyp->learning_rate_decay = rng->gen_uni(10, 120);
    
    hyp->a1_size = rng->gen_uni(2,  128 + 1);
    hyp->a2_size = rng->gen_uni(2,  128 + 1);
    hyp->b1_size = rng->gen_uni(2,  128 + 1);
    hyp->b2_size = rng->gen_uni(2,  128 + 1);

    hyp->dropout = rng->gen_uni_float()*0.5 + 0.5;

    hyp->a1_size /= hyp->dropout;
    hyp->a2_size /= hyp->dropout;
    hyp->b1_size /= hyp->dropout;
    hyp->b2_size /= hyp->dropout;
    
    hyp->l2_reg = rng->gen_uni_float() * 50;
    
    /*int inst_total = hyp->num_instances();
    hyp->batch_size = rng->gen_uni(100, 351);
    hyp->batch_count = inst_total / hyp->batch_size;
    
    hyp->learning_rate = rng->gen_uni_float() * 0.1 + 0.1;
    hyp->learning_rate_decay = rng->gen_uni(10, 61);
    
    hyp->a1_size = rng->gen_uni(20,  48 + 1);
    hyp->a2_size = rng->gen_uni(10,  30 + 1);
    hyp->b1_size = rng->gen_uni( 8, 128 + 1);
    hyp->b2_size = rng->gen_uni( 2,  48 + 1);

    hyp->dropout = rng->gen_uni_float()*0.1 + 0.9;

    hyp->a1_size /= hyp->dropout;
    hyp->a2_size /= hyp->dropout;
    hyp->b1_size /= hyp->dropout;
    hyp->b2_size /= hyp->dropout;
    
    hyp->l2_reg = rng->gen_uni_float()*50;*/
}

void network_grid_search(jup_str data_file) {
    auto hyp = global_options.hyp;
    print_hyperparam(hyp);

    Rng rng {global_rng.rand()};
    
    jout << "Initializing..." << endl;
    // Shut tensorflow up
    setenv("TF_CPP_MIN_LOG_LEVEL", "1", 0);
    init_signal_sigint();
    
    Unique_ptr_free<Training_data> data_train, data_test, data_train_tmp, data_train_tmp2, data_test_tmp;
    network_load_data(data_file, hyp, &data_train, &data_test);
    data_train_tmp  = Training_data::make_unique(data_train->hyp, 32);
    data_train_tmp2 = Training_data::make_unique(data_train->hyp, 32);
    data_test_tmp   = Training_data::make_unique(data_test ->hyp, 32);

    jout << '\n';
    jout << "  trai | test | comp | iter | batch |  rate  | decay | a1 | a2 | b1 | b2 | dropout | l2\n";
    jout.flush();

    double best_loss = std::numeric_limits<double>::infinity();

    Async_shuffler shuffler;
    async_shuffle_init(&shuffler);

    while (true) {
        Hyperparam hyp_rand = data_train->hyp;
        network_random_hyp(&hyp_rand, &rng);

        data_train_tmp ->hyp = hyp_rand;
        data_train_tmp2->hyp = hyp_rand;
        data_test_tmp  ->hyp = hyp_rand;
        data_test_tmp->hyp.batch_count = data_test->hyp.num_instances() / hyp_rand.batch_size;

        async_shuffle_start(&shuffler, data_train.get(), data_train_tmp.get(), 0);
        network_shuffle(data_test.get(), data_test_tmp.get(), 0, true);
        
        auto state = network_init(hyp_rand, true);
        double start_time = elapsed_time();
        int cur_batch = 0;
        
        async_shuffle_wait(&shuffler);
        async_shuffle_start(&shuffler, data_train_tmp.get(), data_train_tmp2.get(), 0);

        while (elapsed_time() < start_time + global_options.grid_max_time) {
            network_batch(state, data_train_tmp->batch(cur_batch), true);
            ++cur_batch;
        
            if (cur_batch >= data_train_tmp->hyp.batch_count) {
                async_shuffle_wait(&shuffler);
                std::swap(data_train_tmp, data_train_tmp2);
                async_shuffle_start(&shuffler, data_train_tmp.get(), data_train_tmp2.get(), 0);
                
                cur_batch = 0;
                ++state->epoch;
                state->epoch_start = state->step;
                
                state->loss_sum1 = 0.f;
                state->loss_count1 = 0;
            }

            if (global_interrupt_flag) break;
        }
        if (global_interrupt_flag) break; // Leak, but we are about to exit anyways
        
        async_shuffle_wait(&shuffler);

        auto vali = network_validate(state, *data_test_tmp);
        double loss_test     = vali.loss_test;
        double loss_test_l2  = vali.loss_test_l2;
        double loss_train    = state->loss_sum1 / state->loss_count1;

        if (loss_test_l2 < best_loss) {
            best_loss = loss_test_l2;
            jout << "* ";
        } else {
            jout << "  ";
        }
        
        jout << jup_printf(
            "%5.3f  %5.3f  %5.3f %6d %7d %8.2e %7d %4d %4d %4d %4d  %8.2e %8.2e",
            loss_train, loss_test, loss_test_l2, state->step, hyp_rand.batch_size,
            (double)hyp_rand.learning_rate, hyp_rand.learning_rate_decay, hyp_rand.a1_size,
            hyp_rand.a2_size, hyp_rand.b1_size, hyp_rand.b2_size, (double)hyp_rand.dropout,
            (double)hyp_rand.l2_reg
        ) << endl;

        network_free(state);
    }
    
    async_shuffle_close(&shuffler);
}

void network_cross_validate(jup_str data_file) {
    auto hyp = global_options.hyp;
    print_hyperparam(hyp);

    if (not global_options.param_in) {
        die("No input parameters specified. It does not make sense to cross-validate a random "
            "model!\nUse the --param-in option to load a parameter checkpoint.");
    }
    
    jout << "Initializing..." << endl;
    // Shut tensorflow up
    setenv("TF_CPP_MIN_LOG_LEVEL", "1", 0);
    init_signal_sigint();

    jout << "  Creating neural network..." << endl;
    auto state = network_init(hyp);

    Unique_ptr_free<Training_data> data_crossval;
    network_load_data(data_file, hyp, &data_crossval);

    jout << "  Restoring from checkpoint... (" << global_options.param_in << ")" << endl;
    network_restore(state);

    auto vali = network_validate(state, *data_crossval);
    jout << "\nResults:\n";
    vali.print(jout);
    jout.flush();

    network_free(state);
}

void network_classify(jup_str graph_file) {
    auto hyp = global_options.hyp;
    print_hyperparam(hyp);

    if (not global_options.param_in) {
        die("No input parameters specified. It does not make sense to classify with a random "
            "model!\nUse the --param-in option to load a parameter checkpoint.");
    }

    jout << "Initializing..." << endl;
    // Shut tensorflow up
    setenv("TF_CPP_MIN_LOG_LEVEL", "1", 0);
    init_signal_sigint();

    jout << "  Using sample count of " << global_options.samples << "..." << endl;
    hyp.batch_count = 1;
    hyp.batch_size = global_options.samples;
    hyp_init_alignment(&hyp);
    auto _data = Training_data::make_unique(hyp);
    auto batch = _data->batch(0);
    jup_memset(&batch.edge_weights);
    jup_memset(&batch.results);

    jout << "  Creating neural network..." << endl;
    auto state = network_init(hyp);

    jout << "  Restoring from checkpoint... (" << global_options.param_in << ")" << endl;
    network_restore(state);
    
    // Need to turn off dropout
    TF_CHECK_OK(state->session.Run({}, {}, {state->dropout_off_op}, nullptr));
    
    Graph_reader_state state_graph;
    graph_reader_init(&state_graph, graph_file);
    Neighbourhood_finder neighbours;

    Timer timer;

    int num_graph = 0;
    int num_positive = 0;
    int num_negative = 0;

    Batch_tensors tensors;
    std::vector<tensorflow::Tensor> outputs;

    jout << "\n   min  |  max  |  avg  | name\n";
    
    while (graph_reader_next(&state_graph)) {
        if (state_graph.graph->num_nodes() == 0) continue;
        
        for (int cur_instance = 0; cur_instance < hyp.batch_size; ++cur_instance) {
            fill_instance(*state_graph.graph, &neighbours, hyp, &batch, cur_instance, false);
        }
        // results are already set to 0

        fill_tensors(hyp, batch, &tensors);
        outputs.clear();
        TF_CHECK_OK(state->session.Run(
            {{state->x, tensors.data_x}},
            {state->y_prob},
            {},
            &outputs
        ));
        
        auto res_yout = outputs[0].vec<float>();

        float min =  std::numeric_limits<float>::infinity();
        float max = -std::numeric_limits<float>::infinity();
        float avg = 0;
        for (int i = 0; i < hyp.batch_size; ++i) {
            if (res_yout(i) < min) min = res_yout(i);
            if (res_yout(i) > max) max = res_yout(i);
            avg += res_yout(i);
        }
        avg /= hyp.batch_size;

        num_positive += avg >  0.f;
        num_negative += avg <= 0.f;

        if (global_interrupt_flag) break;

        jout << jup_printf("  % 6.3f  % 6.3f  % 6.3f  %s", (double)min, (double)max, (double)avg,
                    state_graph.graph->get_name())
             << endl;
        
        ++num_graph;
    }
    
    network_free(state);
    graph_reader_close(&state_graph);

    jout << "Done. (" << timer.total() << ", " << timer.counter_done(num_graph) << " graphs/s)\n\n";

    jout << "Final results:\n"
         << "  Graphs positive: " << num_positive << '\n'
         << "  Graphs negative: " << num_negative << '\n';
}

} /* end of namespace jup */
