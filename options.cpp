
#include "options.hpp"
#include "utilities.hpp"
#include "graph.hpp"
#include "network.hpp"

namespace jup {

Schaf_options global_options;

struct Parse_state {
    Array<jup_str> stack;
    jup_str current;
    jup_str current_option;
};

static bool pop(Parse_state* state) {
    assert(state);

    if (not state->stack) {
        return false;
    } else {
        state->current = state->stack.pop_back();
        return true;
    }
}

[[noreturn]] static void parse_die(Parse_state* state, jup_str message) {
    assert(state);

    jerr << "Error: " << message << '\n';
    if (state->current_option) {
        jerr << "Error: in option " << state->current_option << '\n';
    }
    jerr << "Error: while parsing command line\n";
    jerr << "\nUse the --help option for usage information.\n";
    std::exit(3);
}

static void pop_option_arg(Parse_state* state) {
    assert(state);

    if (not pop(state)) {
        parse_die(state, "Unexpected end of input, expected option argument");
    }
}

static int get_int(
    Parse_state* state,
    int lower = std::numeric_limits<int>::min(),
    int upper = std::numeric_limits<int>::max()
) {
    assert(state);

    int value;
    auto code = jup_stox(state->current, &value);
    if (code) {
        parse_die(state, jup_err_messages[code]);
    } else if (value < lower) {
        parse_die(state, jup_printf("The value is too small, must be at least %d", lower));
    } else if (value > upper) {
        parse_die(state, jup_printf("The value is too big, must be at most %d", upper));
    } else {
        return value;
    }
}

static float get_float(
    Parse_state* state,
    float lower = -std::numeric_limits<float>::infinity(),
    float upper =  std::numeric_limits<float>::infinity()
) {
    assert(state);

    float value;
    auto code = jup_stox(state->current, &value);
    if (code) {
        parse_die(state, jup_err_messages[code]);
    } else if (value < lower) {
        parse_die(state, jup_printf("The value is too small, must be at least %f", lower));
    } else if (value > upper) {
        parse_die(state, jup_printf("The value is too big, must be at most %f", upper));
    } else {
        return value;
    }    
}

static void print_usage() {
    Buffer str;
    str.append(
        "Usage:\n"
        "  schaf [options] [--] mode [args]\n"
        "\n"
        "Modes:\n"
        "  write_graph <input> <output>\n"
        "    Executes the job specified in the jobfile <input>, and writes the resulting graphs "
            "into the file <output>. It is recommended that <output> has the extension "
            "'.schaf.lz4'.\n"
        "\n"
        "  print_stats <input> [output]\n"
        "    Reads the graphs from the file <input> and prints information about them to the "
            "console. If <output> is specified, the information will additionally be written to "
            "that file, in a machine-readable format.\n"
        "\n"
        "  prepare_data <input> <output>\n"
        "    Generates training data for the neural network by reading the graphs in <input> and "
            "writes it into <output>.\n"
        "\n"
        "  train <input>\n"
        "    Read the training data contained in the file <input> and train the network.\n"
        "\n"
        "  print_data_info <input>\n"
        "    Reads the training data from the file <input> and prints information about it to the "
            "console.\n"
        "\n"
        "  dump_graph <input> <output> [index]\n"
        "    Takes a graph from <input> with index <index> and writes a gdf file (as used in "
            "GUESS) describing it. That file can then be displayed by graph-visualisation tools. If "
            "<index> is omitted, the first graph is taken.\n"
        "\n"
        "  dump_graph_random <output> [seed]\n"
        "    Randomly generates a graph and writes a gdf file (see dump_graph) describing it. If "
            "<seed> is omitted, 0 is used.\n"
        "\n"
        "  grid_search <input>\n"
        "    Randomly generates sets of hyperparameters and optimises them for some time. The "
            "results are printed.\n"
        "\n"
        "  cross_validate <input>\n"
        "    Evaluates the network on the specified training data. Specify the parameters to use "
            "via the --param-in option; note that the hyperparameters have to match the saved "
            "network!\n"
        "\n"
        "  classify <input>\n"
        "    Read the graphs from <input> and classify them using the network. Specify the "
            "parameters to use via the --param-in option; note that the hyperparameters have to "
            "match the saved network!\n"
        "\n"
        "Options:\n"
        "  --edges-min,-e <val>  [default: none]\n"
        "  --edges-max,-E <val>  [default: none]\n"
        "    Limits the graphs that are written to graphs with a number of edges inside the "
            "specified range.\n"
        "\n"
        "  --batch-count,-N <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_BATCH_COUNT) "]\n"
        "    Number of batches per training data.\n"
        "\n"
        "  --batch-size,-n <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_BATCH_SIZE) "]\n"
        "    Number of instances per batch.\n"
        "\n"
        "  --recf-nodes <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_RECF_NODES) "]\n"
        "    Number of nodes per receptive field.\n"
        "\n"
        "  --recf-count <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_RECF_COUNT) "]\n"
        "    Number of receptive fields.\n"
        "\n"
        "  --a1-size <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_A1_SIZE) "]\n"
        "  --a2-size <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_A2_SIZE) "]\n"
        "  --b1-size <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_B1_SIZE) "]\n"
        "  --b2-size <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_B2_SIZE) "]\n"
        "    Sizes of the different layers of the neural network.\n"
        "\n"
        "  --gen-instances <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_GEN_INSTANCES) "]\n"
        "    Number of instances generated per graph. Note that these instances may use the same "
            "nodes. Only relevant during mode prepare_data.\n"
        "\n"
        "  --learning-rate,-l <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_LEARNING_RATE) "]\n"
        "    The initial learning rate of the network. Note that when loading a parameter file, "
            "the saved learning rate will be used instead.\n"
        "\n"
        "  --learning-rate-decay,-L <val> [default: 0]\n"
        "    The amount of epochs after which the learning rate is halved. Set to 0 to disable "
            "learning rate decay.\n"
        "\n"
        "  --dropout,-d <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_DROPOUT) "]\n"
        "    The dropout to use for the network, that is the fraction of nodes that is retained "
            "while training. Set to 1.0 to disable dropout.\n"
        "\n"
        "  --l2reg,-2 <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_L2REG) "]\n"
        "    The regularisation strength as applied to the l2 regularisation. Set to 0.f to disable.\n"
        "\n"
        "  --seed,-s <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_SEED) "]\n"
        "    Seed to initialise tensorflow randomness. If set to 0, random randomness is used.\n"
        "\n"
        "  --test-frac <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_TEST_FRAC) "]\n"
        "    The fraction of the data set that is used as test data.\n"
        "\n"
        "  --param-in,-i <path> [default: none]\n"
        "    The parameter file to load. It is used to initialize the networks parameters and the "
            "learning rate.\n"
        "\n"
        "  --iter-max <value> [default: none]\n"
        "    The maximum number of training iterations for the network.\n"
        "\n"
        "  --iter-save <value> [default: " JUP_STRINGIFY(JUP_DEFAULT_ITER_SAVE) "]\n"
        "    The number of iterations after which the parameters will be saved. Set to 0 to "
            "disable saving. The files will be saved in the directory specified by --logdir. If "
            "that options is not set, saving of parameters will also be disabled.\n"
        "\n"
        "  --iter-event <value> [default: " JUP_STRINGIFY(JUP_DEFAULT_ITER_EVENT) "]\n"
        "    The number of iterations after which a summary for tensorboard will be written. Set "
            "to 0 to disable. The files will be saved in the directory specified by --logdir. If "
            "that options is not set, summaries will also be disabled.\n"
        "\n"
        "  --logdir <path> [default: " JUP_DEFAULT_LOGDIR "]\n"
        "    The location to write the summary logfiles (for tensorboard) and the parameter values "
            "to. The directory will be created, if necessary. If this is the empty string, both "
            "logging and saving of parameters are disabled.\n"
        "\n"
        "  --grid-max-time,-T <value> [default: " JUP_STRINGIFY(JUP_DEFAULT_GRID_MAX_TIME) "]\n"
        "    The amount of time a chosen set of hyperparameters (during grid search) is allowed to"
            "optimise, before being terminated.\n"
        "\n"
        "  --grid-params <batch-size> <rate> <a1-size> <a2-size> <b1-size> <b2-size> <dropout> <l2reg>\n"
        "    Set all the hyperparameters at once. Useful for just copy-pasting a grid-search "
            "result. You probably want to set the batch count before this.\n"
        "\n"
        "  --samples,-S <value> [default: " JUP_STRINGIFY(JUP_DEFAULT_SAMPLES) "]\n"
        "    Number of times a neighbourhood will be generated for each graph in mode classify.\n"
        "\n"
        "  --profile <path> [default: none]\n"
        "    Enables profiling. The results will be written to the specified location. Note that "
            "profiling is "
#ifdef USE_PROFILER
            "DISABLED"
#else
            "ENABLED"
#endif
            " in this executable. (Build with USE_PROFILER=1 to enable.)\n"
        "\n"
        "  --help,-h\n"
        "    Prints this message.\n"
        "\n"
    );
    print_wrapped(jerr, str);
}

static bool parse_option(Schaf_options* options, Parse_state* state) {
    if (not pop(state)) {
        parse_die(state, "Unexpected end of input, expected an option or mode.");
    }

    if (not state->current.size() or state->current.front() != '-') {
        return false;
    }
    
    state->current_option = state->current;

    if (
           state->current == "--help" or state->current == "-h"
        or state->current == "-?"     or state->current == "/?"
    ) {
        print_usage();
        std::exit(4);
    } else if (state->current == "--edges-min" or state->current == "-e") {
        pop_option_arg(state);
        options->graph_min_edges = get_int(state, 1);
    } else if (state->current == "--edges-max" or state->current == "-E") {
        pop_option_arg(state);
        options->graph_max_edges = get_int(state, 1);
    } else if (state->current == "--batch-count" or state->current == "-N") {
        pop_option_arg(state);
        options->hyp.batch_count = get_int(state, 1);
    } else if (state->current == "--batch-size" or state->current == "-n") {
        pop_option_arg(state);
        options->hyp.batch_size = get_int(state, 1);
    } else if (state->current == "--recf-nodes") {
        pop_option_arg(state);
        options->hyp.recf_nodes = get_int(state, 1);
    } else if (state->current == "--recf-count") {
        pop_option_arg(state);
        options->hyp.recf_count = get_int(state, 1);
    } else if (state->current == "--gen-instances") {
        pop_option_arg(state);
        options->hyp.gen_instances = get_int(state, 1);
    } else if (state->current == "--a1-size") {
        pop_option_arg(state);
        options->hyp.a1_size = get_int(state, 1);
    } else if (state->current == "--a2-size") {
        pop_option_arg(state);
        options->hyp.a2_size = get_int(state, 1);
    } else if (state->current == "--b1-size") {
        pop_option_arg(state);
        options->hyp.b1_size = get_int(state, 1);
    } else if (state->current == "--b2-size") {
        pop_option_arg(state);
        options->hyp.b2_size = get_int(state, 1);
    } else if (state->current == "--learning-rate" or state->current == "-l") {
        pop_option_arg(state);
        options->hyp.learning_rate = get_float(state, 0.f);
    } else if (state->current == "--learning-rate-decay" or state->current == "-L") {
        pop_option_arg(state);
        options->hyp.learning_rate_decay = get_int(state, 0);
    } else if (state->current == "--dropout" or state->current == "-d") {
        pop_option_arg(state);
        options->hyp.dropout = get_float(state, 0.1f, 1.f);
    } else if (state->current == "--l2reg" or state->current == "-2") {
        pop_option_arg(state);
        options->hyp.l2_reg = get_float(state, 0.f);
    } else if (state->current == "--seed" or state->current == "-s") {
        pop_option_arg(state);
        options->hyp.seed = (u64)get_int(state);
    } else if (state->current == "--test-frac") {
        pop_option_arg(state);
        options->hyp.test_frac = get_float(state, 0.f, 1.0f);
    } else if (state->current == "--param-in" or state->current == "-i") {
        pop_option_arg(state);
        options->param_in = state->current;
    } else if (state->current == "--iter-max") {
        pop_option_arg(state);
        options->iter_max = get_int(state, 0);
    } else if (state->current == "--iter-save") {
        pop_option_arg(state);
        options->iter_save = get_int(state, 1);
    } else if (state->current == "--iter-event") {
        pop_option_arg(state);
        options->iter_event = get_int(state, 0);
    } else if (state->current == "--logdir") {
        pop_option_arg(state);
        options->logdir = state->current;
    } else if (state->current == "--grid-max-time" or state->current == "-T") {
        pop_option_arg(state);
        options->grid_max_time = get_float(state, 0.f);
    } else if (state->current == "--grid-params") {
        int inst_total = options->hyp.num_instances();
        pop_option_arg(state); options->hyp.batch_size = get_int(state, 1);
        options->hyp.batch_count = inst_total / options->hyp.batch_size;
        pop_option_arg(state); options->hyp.learning_rate = get_float(state, 0.f);
        pop_option_arg(state); options->hyp.a1_size = get_int(state, 1);
        pop_option_arg(state); options->hyp.a2_size = get_int(state, 1);
        pop_option_arg(state); options->hyp.b1_size = get_int(state, 1);
        pop_option_arg(state); options->hyp.b2_size = get_int(state, 1);
        pop_option_arg(state); options->hyp.dropout = get_float(state, 0.1f, 1.f);
        pop_option_arg(state); options->hyp.l2_reg = get_float(state, 0.f);
    } else if (state->current == "--profile") {
        pop_option_arg(state);
        options->profiler_loc = state->current;
    } else if (state->current == "--") {
        if (not pop(state)) {
            parse_die(state, "Unexpected end of input, expected a mode.");
        }
        return false;
    } else {
        parse_die(state, "Unknwon option.");
    }

    state->current_option = jup_str {};
    return true;
}

void options_execute(Schaf_options* options, Array_view<jup_str> args) {
    assert(options);
    
    Parse_state state;
    
    for (int i = 0; i < args.size(); ++i) {
        state.stack.push_back(args[args.size() - i - 1]);
    }

    while (true) {
        if (not parse_option(options, &state)) break;
    }

    Profiler_context profiler_context {options->profiler_loc.size(), options->profiler_loc, true};
    
    if (state.current == "write_graph") {
        if (not pop(&state)) {
            parse_die(&state, "Expected the <input> argument to mode write_graph.");
        }
        jup_str input = state.current;
        if (not pop(&state)) {
            parse_die(&state, "Expected the <output> argument to mode write_graph.");
        }
        jup_str output = state.current;
        graph_exec_jobfile(input, output);
    } else if (state.current == "print_stats") {
        if (not pop(&state)) {
            parse_die(&state, "Expected the <input> argument to mode print_stats.");
        }
        jup_str input = state.current;
        jup_str output;
        if (pop(&state)) {
            output = state.current;
        }
        graph_print_stats(input, output);
    } else if (state.current == "prepare_data") {
        if (not pop(&state)) {
            parse_die(&state, "Expected the <input> argument to mode prepare_data.");
        }
        jup_str input = state.current;
        if (not pop(&state)) {
            parse_die(&state, "Expected the <output> argument to mode prepare_data.");
        }
        jup_str output = state.current;
        network_prepare_data(input, output, options->hyp);
    } else if (state.current == "train") {
        if (not pop(&state)) {
            parse_die(&state, "Expected the <input> argument to mode train.");
        }
        jup_str input = state.current;
        network_train(input);
    } else if (state.current == "print_data_info") {
        if (not pop(&state)) {
            parse_die(&state, "Expected the <input> argument to mode print_data_info.");
        }
        jup_str input = state.current;
        network_print_data_info(input);
    } else if (state.current == "dump_graph") {
        if (not pop(&state)) {
            parse_die(&state, "Expected the <input> argument to mode dump_graph.");
        }
        jup_str input = state.current;
        if (not pop(&state)) {
            parse_die(&state, "Expected the <output> argument to mode dump_graph.");
        }
        jup_str output = state.current;
        int index = 0;
        if (pop(&state)) {
            index = get_int(&state, 0);
        }
        graph_dump(input, output, index);
    } else if (state.current == "dump_graph_random") {
        if (not pop(&state)) {
            parse_die(&state, "Expected the <output> argument to mode dump_graph_random.");
        }
        jup_str output = state.current;
        u64 seed = 0;
        if (pop(&state)) {
            seed = (u64)get_int(&state);
        }
        graph_dump_random(output, seed);
    } else if (state.current == "grid_search") {
        if (not pop(&state)) {
            parse_die(&state, "Expected the <input> argument to mode grid_search.");
        }
        jup_str input = state.current;
        network_grid_search(input);
    } else if (state.current == "cross_validate") {
        if (not pop(&state)) {
            parse_die(&state, "Expected the <input> argument to mode cross_validate.");
        }
        jup_str input = state.current;
        network_cross_validate(input);
    } else if (state.current == "classify") {
        if (not pop(&state)) {
            parse_die(&state, "Expected the <input> argument to mode classify.");
        }
        jup_str input = state.current;
        network_classify(input);
    } else {
        auto s = jup_printf(
            "Unknown mode: \"%s\"",
            state.current
        );
        parse_die(&state, s);
    }

    
}

} /* end of namespace jup */
