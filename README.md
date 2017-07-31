# schaf
Statistical Commit History Analysis Framework

## Description

schaf is a tool that generates change graphs from repository metadata and performs statistical analysis of those graphs. It is designed to work with [alarm](https://github.com/suyjuris/alarm), which is an application to download this metadata from GitHub.

## Change graph

A change graph is (currently) defined as follows:

1. Initialize an undirected graph with all files, that exist or have existed in the repository, as nodes. (More precisly, the *paths* of these files.) Set the weight of all edges to 0.
2. Take all commits with exactly one parent (this excludes merges and the initial commit).
3. For each of these commits, compute the set of files that have changed in this commit. Here, changed refers to any change of the blob associated with a path, e.g. an adding, updating or deleting a file would each count as exactly one change, while renaming a file would be two changes, one with the old path and one with the new path.
4. For each pair of nodes in the set of files, increment the weight of the corresponding edge by 1.
5. Delete all edges with weight 0.

As you might note, the number of edges scales quadratically with the largest number of changed files in a commit. While schaf can handle somewhat large graphs (~1e8 edges on my machine with 16 GiB of RAM) you might want to limit the number of edges (using the `--edges-max` option).

## Build instructions

Build the project using a simple `make` . This should work on both the Windows and Linux platform, using a sufficiently new version of `g++` (I have tested it with both 5.4.0 and 6.3.0 .) In the case of Windows, I build using the [mingw64](https://mingw-w64.org/doku.php) project. The tool will not work on big-endian systems. I do some unaligned pointer accesses, which are undefined behaviour and may not work one some architectures (x86_64 should be fine, though).

Building requires `libz` to be installed on the system in a folder `g++` will find. This is the only external dependency.

To build an optimized version of schaf, set the `SCHAF_FAST` environment variable. This will build using `-O3 -march=native -DNDEBUG`, the last of which disables assertions and some checks for allocation behaviour. On my machine this increases the performance by about 5%.

### Notes on compiling on Windows

There is some debugging functionality that is currently only available on the Windows platform, most notably the printing of stacktraces in case of an error. For this to work, the executable has to have debugging information in `.pdb` format. `g++` emits DWARF debugging information, this is converted using [cv2pdb](https://github.com/rainers/cv2pdb). You may acquire the binary however you like, for convenience's sake the Makefile downloads it into `/usr/local/bin` via `make init` . Additionally, this makes it possible to debug the executable using MSVS, but `gdb` of course does not work with `.pdb` data. For this purpose the original executable is left in the `build_files/` subfolder, just do `gdb build_files/schaf.exe` .

## Libraries

schaf makes use of the following libraries:

* [lz4](http://www.lz4.org/), written by Yann Collet
* [SHA-1 in C](https://github.com/clibs/sha1), written by Steve Reid
* [sparsehash](https://github.com/sparsehash/sparsehash), written at Google Inc.
* [StackWalker](https://stackwalker.codeplex.com/), written by Jochen Kalmbach
* [xxHash](http://www.xxhash.com/), written by Yann Collet
* [zlib](http://zlib.net/), written by Jean-loup Gailly and Mark Adler

zlib is an external dependency, the others can be found in the `libs/` directory.

## Usage Information

    $ ./schaf --help
    Usage:
      schaf [options] [--] mode [args]

    Modes:
      write_graph <input> <output>
        Executes the job specified in the jobfile <input>, and writes the resulting
        graphs into the file <output>. It is recommended that <output> has the
        extension '.schaf.lz4'.

      print_stats <input> [output]
        Reads the graphs from the file <input> and prints information about them to
        the console. If <output> is specified, the information will additionally be
        written to that file, in a machine-readable format.

    Options:
      --edges-min,-e <val>  [default: none]
      --edges-max,-E <val>  [default: none]
        Limits the graphs that are written to graphs with a number of edges inside
        the specified range.

      --help,-h
        Prints this message.

## File format

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
        
      size is an u32, specifying the number of items (not the number of
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
      details of the hash maps used.

