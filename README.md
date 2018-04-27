# schaf
Statistical Commit History Analysis Framework

## Description

`schaf` is a tool that generates change graphs from repository metadata and performs statistical analysis of those graphs. It is designed to work with [alarm](https://github.com/suyjuris/alarm), which is an application to download this metadata from GitHub.

In particular, I am interested in estimating how realistic a certain change-coupling is. To this end, I train a binary classifier based on CNNs to differentiate between real and corrupted / randomly generated data.

For more details, see my thesis ([html](https://nicze.de/philipp/bsc_thesis.html), [pdf](https://nicze.de/philipp/bsc_thesis.pdf)).

Feel free to contact me at <contact@nicze.de>.

## Change-coupling graph

A change-coupling graph is defined as follows:

1. Initialize an undirected graph with all files, that exist or have existed in the repository, as nodes. (More precisly, the *paths* of these files.) Set the weight of all edges to 0.
2. Take all commits with exactly one parent (this excludes merges and the initial commit).
3. For each of these commits, compute the set of files that have changed in this commit. Here, changed refers to any change of the blob associated with a path, e.g. an adding, updating or deleting a file would each count as exactly one change, while renaming a file would be two changes, one with the old path and one with the new path.
4. For each pair of nodes in the set of files, increment the weight of the corresponding edge by 1.
5. Delete all edges with weight 0.

As you might note, the number of edges scales quadratically with the largest number of changed files in a commit. While `schaf` can handle somewhat large graphs (~1e8 edges on my machine with 16 GiB of RAM) you might want to limit the number of edges (using the `--edges-max` option).

## Libraries

`schaf` makes use of the following libraries:

* [lz4](http://www.lz4.org/), written by Yann Collet
* [SHA-1 in C](https://github.com/clibs/sha1), written by Steve Reid
* [sparsehash](https://github.com/sparsehash/sparsehash), developed at Google
* [StackWalker](https://stackwalker.codeplex.com/), written by Jochen Kalmbach
* [xxHash](http://www.xxhash.com/), written by Yann Collet
* [zlib](http://zlib.net/), written by Jean-loup Gailly and Mark Adler
* [TensorFlow](https://www.tensorflow.org/), developed at Google

zlib and TensorFlow are external dependencies, the others can be found in the `libs/` directory.

## Build instructions

After you installed the requirements, set up the repository via `make init` and build the project using a simple `make`, which will produce an executable called `schaf`. This should work on a 64-bit little-endian Linux system, using a sufficiently new version of `g++` (I have tested it with both 6.3.0 .) The tool will not work on big-endian systems, it _may_ work on 32-bit platforms (but I take no responsibility). I do some unaligned pointer accesses, which are undefined behaviour and may not work one some architectures or compilers (x86_64 should be fine, though).

To build an optimized version of `schaf`, set the `SCHAF_FAST` environment variable. This will build using `-O3 -march=native -DNDEBUG`, the last of which disables assertions and some checks for allocation behaviour. On my machine this increases performance significantly, although it does not matter that much when training the NN.

Building requires `libz`, `libtensorflow_cc` and `tensorflow_framework` (TensorFlow r1.5) to be installed on the system in a folder `g++` will find. Additionally, `pthread`s have to be available.

Linking a C++ application to TensorFlow was kind of a mess when I started this (maybe still is), so there are some hoops I had to jump through. If you want to get it to work the way I did, refer to the next section and you should get something running. In case you want to use a newer version of TensorFlow or already have some things set up and do not want to start from scratch, there are some notes in the subsequent section to help you out.

### The tested way to build

First, install the prerequisites mentioned in the TensorFlow documentation for compiling from source, **for version 1.5**. Should be [here](https://www.tensorflow.org/versions/r1.5/install/install_sources), but I have taken the liberty of reproducing the essential step in the following.

Basically, you need some Python packages (`sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel`) and `bazel` (see [here](https://docs.bazel.build/versions/master/install.html)). If want to use your NVIDIA GPU, you'll need NVIDIA's Cuda Toolkit (>= 7.0, recommended is 9.0), the associated drivers, and cuDNN (>= 3.0, recommended is 7.0). See the (NVIDIA documentation)[http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A] for details.

Still with me? Great. Now we'll get to the actual TensorFlow sources.

    git clone -b v1.5.0 https://github.com/tensorflow/tensorflow.git tensorflow
    cd tensorflow
    ./configure

This script will ask you a bunch of questions, and you'll have to do your best to answer them.
* The Python path you specify should match the one you installed the packages for. In particular, use `/usr/bin/python3` if you just copied my command above.
* If you went through the trouble of installing CUDA, you should enable that
* CUDA 9.0 (and 9.1) does not play well with newer version of gcc, so make it use the older one if necessary. Look up compatibility in the (NVIDIA documentation)[http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A] (should be Table 1).

For me, it looked like this:

    Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3
    Please input the desired Python library path to use.  Default is [/usr/lib/python3.6/dist-packages]
    Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: 
    Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
    Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
    Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
    Do you wish to build TensorFlow with XLA JIT support? [y/N]: 
    Do you wish to build TensorFlow with GDR support? [y/N]: 
    Do you wish to build TensorFlow with VERBS support? [y/N]: 
    Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 
    Do you wish to build TensorFlow with CUDA support? [y/N]: y
    Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 9.1
    Please specify the location where CUDA 9.1 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
    Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 
    Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
    Please specify a list of comma-separated Cuda compute capabilities you want to build with.
    Do you want to use clang as CUDA compiler? [y/N]: 
    Do you wish to build TensorFlow with MPI support? [y/N]: 
    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 

Now, lets build.

    bazel build --config=opt --config=cuda //tensorflow:libtensorflow_cc.so

This takes a while. Afterwards, there should be both `libtensorflow_cc.so` and `libtensorflow_framework.so` in `bazel-bin/tensorflow/`, those need to be installed.

    sudo cp bazel-bin/tensorflow/libtensorflow_cc.so bazel-bin/tensorflow/libtensorflow_framework.so /usr/local/lib/

And we're done. Going back to the directory of `schaf` and issuing `make init` followed by `make` should work now. To test the application, run `schaf --help`—if that works then you are probably fine.

### Notes for brave souls

So you have decided that just getting the version of TensorFlow I tested with to run is not adventurous enough for you? Well, your choice. For `schaf` to compile, you basically need two things: The headers for TensorFlow's C++ API, and the corresponding shared libraries. Note that **TensorFlow's C++ API is not the same as its C API and the C++ symbols may very well be missing from prebuilt binaries**. To add to the fun, be careful when mixing C++ code compiled by compilers that are not exactly the same.

Also, the headers obviously need to match the version that was compiled for the binaries. To make things more interesting, there is (to the extend of my knowledge) no easy way to get all the necessary headers out of the TensorFlow sources. You may find the `libs/copy_headers.sh` script useful, which copies the relevant ones (for r1.5, but maybe those have not changed) over. `schaf` has headers for TensorFlow r1.5 in its repository, in `libs/tensorflow_linux.tar.bz2` which are added to the local include directory in `build_files/include_linux` when you do `make init`. So delete those and add your own headers somewhere in the include path.

The last few times I upgraded TensorFlow it went through without much trouble, but you should be prepared to fix some minor API changes in my code, or open a ticket and I'll see what I can do.

## Usage Information

    $ ./schaf --help
    Usage:
      schaf [options] [--] mode [args]
    
    Modes:
      write_graph <input> <output>
        Executes the job specified in the jobfile <input>, and writes the resulting graphs into the file
        <output>. It is recommended that <output> has the extension '.schaf.lz4'.
    
      print_stats <input> [output]
        Reads the graphs from the file <input> and prints information about them to the console. If
        <output> is specified, the information will additionally be written to that file, in a
        machine-readable format.
    
      prepare_data <input> <output>
        Generates training data for the neural network by reading the graphs in <input> and writes it
        into <output>.
    
      train <input>
        Read the training data contained in the file <input> and train the network.
    
      print_data_info <input>
        Reads the training data from the file <input> and prints information about it to the console.
    
      dump_graph <input> <output> [index]
        Takes a graph from <input> with index <index> and writes a gdf file (as used in GUESS)
        describing it. That file can then be displayed by graph-visualisation tools. If <index> is
        omitted, the first graph is taken.
    
      dump_graph_random <output> [seed]
        Randomly generates a graph and writes a gdf file (see dump_graph) describing it. If <seed> is
        omitted, 0 is used.
    
      grid_search <input>
        Randomly generates sets of hyperparameters and optimises them for some time. The results are
        printed.
    
      cross_validate <input>
        Evaluates the network on the specified training data. Specify the parameters to use via the
        --param-in option; note that the hyperparameters have to match the saved network!
    
      classify <input>
        Read the graphs from <input> and classify them using the network. Specify the parameters to use
        via the --param-in option; note that the hyperparameters have to match the saved network!
    
    Options:
      --edges-min,-e <val>  [default: 10]
      --edges-max,-E <val>  [default: 50000000]
        Limits the graphs that are written to graphs with a number of edges inside the specified range.
    
      --batch-count,-N <val> [default: 256]
        Number of batches per training data.
    
      --batch-size,-n <val> [default: 256]
        Number of instances per batch.
    
      --recf-nodes <val> [default: 8]
        Number of nodes per receptive field.
    
      --recf-count <val> [default: 8]
        Number of receptive fields.
    
      --a1-size <val> [default: 24]
      --a2-size <val> [default: 32]
      --b1-size <val> [default: 72]
      --b2-size <val> [default: 32]
        Sizes of the different layers of the neural network.
    
      --gen-instances <val> [default: 32]
        Number of instances generated per graph. Note that these instances may use the same nodes. Only
        relevant during mode prepare_data.
    
      --learning-rate,-l <val> [default: 0.1]
        The initial learning rate of the network. Note that when loading a parameter file, the saved
        learning rate will be used instead.
    
      --learning-rate-decay,-L <val> [default: 0]
        The amount of epochs after which the learning rate is halved. Set to 0 to disable learning rate
        decay.
    
      --dropout,-d <val> [default: 1.0]
        The dropout to use for the network, that is the fraction of nodes that is retained while
        training. Set to 1.0 to disable dropout.
    
      --l2reg,-2 <val> [default: 0.0]
        The regularisation strength as applied to the l2 regularisation. Set to 0.f to disable.
    
      --seed,-s <val> [default: 1]
        Seed to initialise tensorflow randomness. If set to 0, random randomness is used.
    
      --test-frac <val> [default: 0.1]
        The fraction of the data set that is used as test data.
    
      --param-in,-i <path> [default: none]
        The parameter file to load. It is used to initialize the networks parameters and the learning
        rate.
    
      --iter-max <value> [default: none]
        The maximum number of training iterations for the network.
    
      --iter-save <value> [default: 5000]
        The number of iterations after which the parameters will be saved. Set to 0 to disable saving.
        The files will be saved in the directory specified by --logdir. If that options is not set,
        saving of parameters will also be disabled.
    
      --iter-event <value> [default: 1000]
        The number of iterations after which a summary for tensorboard will be written. Set to 0 to
        disable. The files will be saved in the directory specified by --logdir. If that options is not
        set, summaries will also be disabled.
    
      --logdir <path> [default: tf_data]
        The location to write the summary logfiles (for tensorboard) and the parameter values to. The
        directory will be created, if necessary. If this is the empty string, both logging and saving of
        parameters are disabled.
    
      --grid-max-time,-T <value> [default: 30.0]
        The amount of time a chosen set of hyperparameters (during grid search) is allowed to optimise,
        before being terminated.
    
      --grid-params <batch-size> <rate> <decay> <a1-size> <a2> <b1> <b2> <dropout> <l2reg>
        Set all the hyperparameters at once. Useful for just copy-pasting a grid-search result. You
        probably want to set the batch count before this.
    
      --samples,-S <value> [default: 8]
        Number of times a neighbourhood will be generated for each graph in mode classify.
    
      --profile <path> [default: none]
        Enables profiling. The results will be written to the specified location. Note that profiling is
        ENABLED in this executable. (Build with USE_PROFILER=1 to enable.)
    
      --help,-h
        Prints this message.

## File format for graphs

This is the format `schaf` uses to store change-coupling graphs on disk, and the only format it can currently read. If you have trouble outputting those and would like `schaf` to support a different format, please let me know.

An `u32` is a litte-endian 32-bit unsigned integer. All numbers are `u32`s. The
file consists of binary data, in the following structure:

    - A header appears at the beginning, consisting of:
    
        4-byte magic number: "^<k\x85"
        
    - Some number of graphs follow. They start with a header:
    
        size_uncompressed size_compressed
        
      (Both are u32 numbers, as are all other numbers.) Then, exactly
      size_compressed bytes of data follow, which are LZ4 compressed and decompress
      into exactly size_uncompressed bytes. (Note that this uses only the basic LZ4
      compression, not the LZ4 frame format.) They have the following structure:
      
        offset_name offset_nodes offset_edges ...
        
      Each of these are to be interpreted as a relative offset to the beginning of
      the respective u32 in bytes. (If, for example, the first byte of offset_nodes
      was at position 4, which it should be, and holds the value 24, then the first
      byte of the nodes structure would be at position 28.) offset_name should
      always hold the value 12. These three offsets each specify the location of an
      array, with the structure
        
        size item_1 item_2 ... item_size
        
      size is a u32, specifying the number of items (not necessarily the number of
      bytes!). Then, exactly size items follow.
      
      offset_name points to an array of bytes, which are a zero-terminated string
      containing the name of the repository, in the form <owner>/<repo>
      (e.g. suyjuris/schaf).
      
      offset_nodes points to an array of u32s, with n+1 items, n being the number of
      nodes. These are offsets into the offset_edges array, in the unit items (not
      bytes!). The first offset is always 0, the last is always one past the end of
      the offset_edges array, and they are (not necessarily strictly) monotonically
      increasing.
      
      offset_edges points to an array of pairs of u32, each item having the
      following structure:
      
        node_other weight
        
      The offsets in the nodes array (the one pointed to by offset_nodes)
      effectively define a range of items for each node. More specifically, the
      edges of node i begin at the offset nodes[i] and ends one before the offset
      nodes[i+1]. The items (with edges being the array pointed to by offset_edges)
      
        edges[node[i]] edges[node[i] + 1] ... edges[node[i+1] - 1]
      
      are the adjacency list of node i. Each item contains the id of an adjacent
      node, node_other, and the weight of the edge. (The id of a node is simply its
      index in the nodes array, a number from 0 to n-1, n being the number of
      nodes.)
      
      Note that the number of nodes is the size of the nodes array - 1, and the
      number of edges is the size of the edges array / 2.
      
      The arrays name, nodes and edges are written in exactly this order (meaning
      that offset_name < offset_nodes < offset_edges holds), with no padding bytes
      in between.
      
      The format specified above is unique, as long a certain order of nodes is
      given. schaf is deterministic, but the order of nodes depends on implementation
      details of the hash maps used and may vary on different systems.

