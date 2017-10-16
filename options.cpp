
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
        "  --batch-nodes <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_BATCH_NODES) "]\n"
        "    Number of nodes per instance.\n"
        "\n"
        "  --gen-graph-nodes <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_GEN_GRAPH_NODES) "]\n"
        "    Number of nodes needed per instance. For example, when choosing a value of 32, a "
            "graph with 256 nodes would be used to generate 8 instances. However, these instances "
            "may use the same nodes.\n"
        "\n"
        "  --learning-rate,-l <val> [default: " JUP_STRINGIFY(JUP_DEFAULT_LEARNING_RATE) "]\n"
        "    The initial learning rate of the network. Note that when loading a parameter file, "
            "the saved learning rate will be used instead.\n"
        "\n"
        "  --param-in,-i <path> [default: none]\n"
        "    The parameter file to load. It is used to initialize the networks parameters and the "
            "learning rate.\n"
        "\n"
        "  --param-out,-o <path> [default: none]\n"
        "    The parameter file to save. The parameter of the network will be saved to this "
            "location.\n"
        "\n"
        "  --iter-max <value> [default: none]\n"
        "    The maximum number of training iterations for the network.\n"
        "\n"
        "  --iter-save <value> [default: " JUP_STRINGIFY(JUP_DEFAULT_ITER_SAVE) "]\n"
        "    The number of iterations after which the parameters will be saved.\n"
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
    } else if (state->current == "--batch-nodes") {
        pop_option_arg(state);
        options->hyp.batch_nodes = get_int(state, 1);
    } else if (state->current == "--gen-graph-nodes") {
        pop_option_arg(state);
        options->hyp.gen_graph_nodes = get_int(state, 1);
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
    } else {
        auto s = jup_printf(
            "Unknown mode. Expected write_graph, print_stats or prepare_data, got %s",
            state.current
        );
        parse_die(&state, s);
    }

    
}

} /* end of namespace jup */
