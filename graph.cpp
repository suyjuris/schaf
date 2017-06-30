
#include "libs/sparsehash.hpp"
#include "libs/lz4.hpp"
#include "libs/xxhash.hpp"

#include "allocator.hpp"
#include "array.hpp"
#include "graph.hpp"
#include "debug.hpp"
#include "utilities.hpp"

namespace jup {

constexpr static jup_str SCHAFFILE_MAGIC = "^<k\x85";

using Path_hash_t = u64;

template <int n>
static constexpr inline u64 rotate_left (u64 x) {
    static_assert(0 < n and n < 64);
    return (x << n) | (x >> (64-n));
}


constexpr static u64 PRIME64_1 = 11400714785074694791ULL;
constexpr static u64 PRIME64_2 = 14029467366897019727ULL;
constexpr static u64 PRIME64_3 =  1609587929392839161ULL;
constexpr static u64 PRIME64_4 =  9650029242287828579ULL;
constexpr static u64 PRIME64_5 =  2870177450012600261ULL;

static Path_hash_t concatenate_path(Path_hash_t a, u32 b) {
    // XXHash, specialized for an u64 and an u32
    // see https://github.com/Cyan4973/xxHash
    u64 h64 = PRIME64_5 + 12;
    h64 ^= rotate_left<31>(a * PRIME64_2) * PRIME64_1;
    h64  = rotate_left<27>(h64) * PRIME64_1 + PRIME64_4;
    h64 ^= (u64)b * PRIME64_1;
    h64  = rotate_left<23>(h64) * PRIME64_2 + PRIME64_3;
    h64  = (h64 ^ (h64 >> 33)) * PRIME64_2;
    h64  = (h64 ^ (h64 >> 29)) * PRIME64_3;
    h64 ^= h64 >> 32;
    return h64;
}

using Edge_t = u64;

struct Hasher_Edge_t {
    using argument_type = Edge_t;
    using result_type = std::size_t;
    
    std::size_t operator() (Edge_t val) const noexcept {
        // splitmix64, see http://xorshift.di.unimi.it/splitmix64.c
        val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9ull;
        //val = (val ^ (val >> 27)) * 0x94d049bb133111ebull;
        val = val ^ (val >> 31);
        return val;
    }
};

template <u8 offset, u8 shift>
void radix_pass_lsb(u32 const* aux, u32* into, u32 size, u32* first) {
    for (u32 i = 0; i < size; i += 3) {
        u32 pos = 3 * first[(aux[i + offset] >> shift) & 0xff]++;
        into[pos  ] = aux[i  ]; // node_a
        into[pos+1] = aux[i+1]; // node_b
        into[pos+2] = aux[i+2]; // weight
    }
}
template <u8 offset, u8 shift>
void radix_pass_lsb_last(u32 const* aux, u32* into, u32 size, u32* first, u32* offsets) {
    for (u32 i = 0; i < size; i += 3) {
        u32 pos = 2 * first[(aux[i + offset] >> shift) & 0xff]++;
        offsets[aux[i] + 1] = pos/2+1;
        into[pos  ] = aux[i+1]; // node_b
        into[pos+1] = aux[i+2]; // weight
    }
}

using Map_commits_t = google::sparse_hash_map<Sha_t, Git_commit const*>;
using Map_trees_t = google::sparse_hash_map<Sha_t, Git_tree const*>;
using Map_nodes_t = google::dense_hash_map<Path_hash_t, u32>;
using Map_edges_t = google::dense_hash_map<Edge_t, u32, Hasher_Edge_t>;

template <typename T>
void hash_map_init(T* map, typename T::key_type empty) {}

template <typename K, typename V, typename H, typename K_>
void hash_map_init(google::dense_hash_map<K, V, H>* map, K_ empty) {
    map->set_empty_key(empty);
}

static void add_recursive(
    Git_tree_Entry const& entry,
    Path_hash_t hash_prefix,
    Map_trees_t const& trees,
    Array<Path_hash_t>* changed,
    int arg
) {
    Path_hash_t hash = concatenate_path(hash_prefix, entry.name);
                    
    //jdbg < arg < hash ,0;
    if (entry.mode == Git_tree_Entry::DIR) {
        for (auto const& i: trees.find(entry.sha)->second->entries) 
            add_recursive(i, hash, trees, changed, 9);
    } else {
        changed->push_back(hash);
    }
}

static void calculate_diff(
    Path_hash_t hash_prefix,
    Git_tree const& tree1,
    Git_tree const& tree2,
    Map_trees_t const& trees,
    Array<Path_hash_t>* changed
) {
    assert(changed);
    
    int i = 0, j = 0;
    int l1 = tree1.entries.size();
    int l2 = tree2.entries.size();

    while (true) {
        if (j == l2) {
            for (; i < l1; ++i) {
                // The file was deleted
                add_recursive(tree1.entries[i], hash_prefix, trees, changed, 1);
            }
            break;
        }
        
        {u32 jname = tree2.entries[j].name;
        for (; i < l1 and tree1.entries[i].name < jname; ++i) {
            // The file was deleted
            add_recursive(tree1.entries[i], hash_prefix, trees, changed, 2);
        }}
        
        if (i == l1) {
            for (; j < l2; ++j) {
                // The file was created
                add_recursive(tree2.entries[j], hash_prefix, trees, changed, 3);
            }
            break;
        }
        
        {u32 iname = tree1.entries[i].name;
        for (; j < l2 and tree2.entries[j].name < iname; ++j) {
            // The file was deleted
            add_recursive(tree2.entries[j], hash_prefix, trees, changed, 4);
        }}

        if (j == l2) {
            for (; i < l1; ++i) {
                // The file was deleted
                add_recursive(tree1.entries[i], hash_prefix, trees, changed, 5);
            }
            break;
        }
        
        while (true) {
            auto const& tree1_el = tree1.entries[i];
            auto const& tree2_el = tree2.entries[j];
            if (tree1_el.name != tree2_el.name) break;
            
            if (tree1_el.sha != tree2_el.sha) {
                bool is_t1 = tree1_el.mode == Git_tree_Entry::DIR;
                bool is_t2 = tree2_el.mode == Git_tree_Entry::DIR;
                Path_hash_t hash = concatenate_path(hash_prefix, tree1_el.name);
                if (is_t1 and is_t2) {
                    // Somewhere in this directory something has changed
                    calculate_diff(hash, *trees.find(tree1_el.sha)->second,
                        *trees.find(tree2_el.sha)->second, trees, changed);
                } else if (is_t1 or is_t2) {
                    // A file was replaced by a directory or vice versa
                    add_recursive(tree1_el, hash_prefix, trees, changed, 6);
                    add_recursive(tree2_el, hash_prefix, trees, changed, 7);
                } else {
                    // The file was changed
                    //jdbg < 8 < hash ,0;
                    changed->push_back(hash);
                }
            }
            ++i; ++j;
            if (i == l1 or j == l2) break;
        }
    }
}

void graph_generate_single(Alarm_stream* stream, std::ostream* out) {
    Map_commits_t commits;
    Map_trees_t trees;
    Arena_allocator arena;
    
    while (not alarm_parse_eof(stream)) {
        alarm_progress(stream);
        auto& objects = alarm_parse(stream);
        assert(objects.size());
        arena.store(&stream->out_data);
        
        for (auto& i: objects) {
            if (auto commit = i.as_commit()) {
                commits[commit->sha] = commit;
            } else if (auto tree = i.as_tree()) {
                trees[tree->sha] = tree;
                std::sort(tree->entries.begin(), tree->entries.end());
            } else {
                assert(false);
            }
        }
    }

    jout << "Parsing complete (commits: " << stream->num_commits << ", trees: " << stream->num_trees
         << "), starting with graph generation..." << endl;
    
    Map_nodes_t nodes;
    Map_edges_t edges;
    Array<Path_hash_t> changed;

    hash_map_init(&nodes, -1);
    hash_map_init(&edges, 0); // 0 is fine, because the edge (0, 0) can not exist

    {
    int count_commits = 0;
    int count_edges  = 0;

    auto start_t = std::clock();
    auto beg_c   = start_t;
    auto beg_t   = std::time(nullptr);

    /*
    constexpr static int hist_changed_size = 20;
    int hist_changed[32] = {0};
    */
    
    for (auto const& it: commits) {
        if (it.second->parents.size() != 1) continue;
        
        auto now_t = std::time(nullptr);
        if (std::difftime(now_t, beg_t) >= 5) {
            beg_t = now_t;
            float f = (float)count_commits / (float)commits.size() * 100.f;

            auto now_c = std::clock();
            float elapsed = (float)(now_c - beg_c) / (float)CLOCKS_PER_SEC;
            float speed = (float)count_edges / elapsed;
            count_edges = 0;
            beg_c = now_c;
            
            jout << jup_printf("Generating graph... (%5.2f%%, %.0f incr/s)", f, speed) << endl;
        }
        ++count_commits;

        changed.reset();
        
        Git_tree const& tree1 = *trees.find(it.second->tree)->second;
        Git_tree const& tree2 = *trees.find(commits.find(it.second->parents[0])->second->tree)->second;
        
        calculate_diff(0, tree1, tree2, trees, &changed);

        int n = changed.size();
        count_edges += n*(n-1) / 2;

        /*
        if (n) {
            int i = 31 - __builtin_clz(n);
            ++hist_changed[i];
        }
        */
        
        if (n > 1000 and (int)edges.size() < n*(n-1) / 2) {
            edges.resize(n*(n-1) / 2);
        }

        for (auto& i: changed) {
            i = nodes.insert({i, nodes.size()}).first->second;
        }
        std::sort(changed.begin(), changed.end());

        for (int i = 0; i < changed.size(); ++i) {
            u64 node_i = changed[i] << 32;
            
            for (int j = 0; j < i; ++j) {
                u64 node_j = changed[j];

                edges[node_i | node_j] += 1;
            }
        }
    }

    float f = (float)(std::clock() - start_t) / (float)CLOCKS_PER_SEC;
    jout << jup_printf("Finished in %.2fs.\n", f);
    jout << "The graph has " << nodes.size() << " nodes and " << edges.size() << " edges." << endl;
    }


    /*
    jout << "Histogram of commit change count:\n  exp      n_changes\n";
    for (int i = 0; i < hist_changed_size; ++i) {
        jout << jup_printf("  %3d %12d\n", i, hist_changed[i]);
    }
    jout.flush();

    
    constexpr static int hist_edge_size = 16;
    int hist_edge[hist_edge_size] = {0};
    for (auto const& i: edges) {
        if (i.second <= hist_edge_size) ++hist_edge[i.second - 1];
    }
    jout << "Histogram of edge weights:\n  val      n_nodes\n";
    for (int i = 0; i < hist_edge_size; ++i) {
        jout << jup_printf("  %3d %12d\n", i+1, hist_edge[i]);
    }
    jout.flush();
    */

    Buffer graph_data;
    {
    auto start_t = std::clock();
    jout << "Packing graph... ";
    jout.flush();

    u32 aux_size = edges.size() * 6;
    u32* aux1 = new u32[aux_size];
    u32* aux2 = new u32[aux_size];
    u32* last_0  = new u32[256] {0};
    u32* last_1  = new u32[256] {0};
    u32* last_2  = new u32[256] {0};
    u32* last_3  = new u32[256] {0};

    if (nodes.size() > 0xffff) {
        {int i = 0;
        for (auto it: edges) {
            u32 node_a = (u32)(it.first >> 32);
            u32 node_b = (u32)(it.first);
            u32 weight = it.second;

            ++last_0[ node_a        & 0xff];
            ++last_1[(node_a >>  8) & 0xff];
            ++last_2[(node_a >> 16) & 0xff];
            ++last_3[(node_a >> 24) & 0xff];
            ++last_0[ node_b        & 0xff];
            ++last_1[(node_b >>  8) & 0xff];
            ++last_2[(node_b >> 16) & 0xff];
            ++last_3[(node_b >> 24) & 0xff];
            aux1[i++] = node_a;
            aux1[i++] = node_b;
            aux1[i++] = weight;
            aux1[i++] = node_b;
            aux1[i++] = node_a;
            aux1[i++] = weight;
        }}

        edges.clear();

        for (int j = 1; j < 256; ++j) {
            last_0[j] += last_0[j-1];
            last_1[j] += last_1[j-1];
            last_2[j] += last_2[j-1];
            last_3[j] += last_3[j-1];
        } 

        u32* first_0 = new u32[256];
        u32* first_1 = new u32[256];
        u32* first_2 = new u32[256];
        u32* first_3 = new u32[256];

        first_0[0] = 0; std::memcpy(first_0 + 1, last_0, sizeof(u32)*255);
        first_1[0] = 0; std::memcpy(first_1 + 1, last_1, sizeof(u32)*255);
        first_2[0] = 0; std::memcpy(first_2 + 1, last_2, sizeof(u32)*255);
        first_3[0] = 0; std::memcpy(first_3 + 1, last_3, sizeof(u32)*255);

        radix_pass_lsb<1, 0>(aux1, aux2, aux_size, first_0);
        radix_pass_lsb<1, 8>(aux2, aux1, aux_size, first_1);  
        radix_pass_lsb<1,16>(aux1, aux2, aux_size, first_2);  
        radix_pass_lsb<1,24>(aux2, aux1, aux_size, first_3);

        first_0[0] = 0; std::memcpy(first_0 + 1, last_0, sizeof(u32)*255);
        first_1[0] = 0; std::memcpy(first_1 + 1, last_1, sizeof(u32)*255);
        first_2[0] = 0; std::memcpy(first_2 + 1, last_2, sizeof(u32)*255);
        first_3[0] = 0; std::memcpy(first_3 + 1, last_3, sizeof(u32)*255);

        radix_pass_lsb<0, 0>(aux1, aux2, aux_size, first_0);
        radix_pass_lsb<0, 8>(aux2, aux1, aux_size, first_1);  
        radix_pass_lsb<0,16>(aux1, aux2, aux_size, first_2);  

        graph_data.take(aux1, aux_size);
        graph_data.reserve(Graph::total_space(nodes.size(), aux_size / 6));
        Graph& g = graph_data.emplace_back<Graph>();

        g.nodes.init((u32)nodes.size() + 1, &graph_data);
        g.edge_data.init(aux_size / 3, &graph_data);
        assert(g.num_nodes() == (int)nodes.size());
        assert(g.num_edges() == (int)aux_size / 6);

        static_assert(sizeof(Edge) == sizeof(u64));
        u32* node_data = (u32*)g.nodes.begin();
        u32* edge_data = (u32*)g.edge_data.begin();

        radix_pass_lsb_last<0,24>(aux2, edge_data, aux_size, first_3, node_data);
      
        node_data[0] = 0;
        for (u32 i = 1; i < g.nodes.size(); ++i) {
            if (node_data[i] == 0) g.nodes[i] = g.nodes[i-1];
        }
 
        delete[] first_0;
        delete[] first_1;
        delete[] first_2;
        delete[] first_3;
    } else {    
        {int i = 0;
        for (auto it: edges) {
            u32 node_a = (u32)(it.first >> 32);
            u32 node_b = (u32)(it.first);
            u32 weight = it.second;

            ++last_0[ node_a        & 0xff];
            ++last_1[(node_a >>  8) & 0xff];
            ++last_0[ node_b        & 0xff];
            ++last_1[(node_b >>  8) & 0xff];
            aux1[i++] = node_a;
            aux1[i++] = node_b;
            aux1[i++] = weight;
            aux1[i++] = node_b;
            aux1[i++] = node_a;
            aux1[i++] = weight;
        }}

        edges.clear();

        for (int j = 1; j < 256; ++j) {
            last_0[j] += last_0[j-1];
            last_1[j] += last_1[j-1];
        } 

        u32* first_0 = new u32[256];
        u32* first_1 = new u32[256];

        first_0[0] = 0; std::memcpy(first_0 + 1, last_0, sizeof(u32)*255);
        first_1[0] = 0; std::memcpy(first_1 + 1, last_1, sizeof(u32)*255);

        radix_pass_lsb<1, 0>(aux1, aux2, aux_size, first_0);
        radix_pass_lsb<1, 8>(aux2, aux1, aux_size, first_1);  

        first_0[0] = 0; std::memcpy(first_0 + 1, last_0, sizeof(u32)*255);
        first_1[0] = 0; std::memcpy(first_1 + 1, last_1, sizeof(u32)*255);

        radix_pass_lsb<0, 0>(aux1, aux2, aux_size, first_0);

        graph_data.take(aux1, aux_size);
        graph_data.reserve(Graph::total_space(nodes.size(), aux_size / 6));
        Graph& g = graph_data.emplace_back<Graph>();

        g.nodes.init((u32)nodes.size() + 1, &graph_data);
        g.edge_data.init(aux_size / 3, &graph_data);
        assert(g.num_nodes() == (int)nodes.size());
        assert(g.num_edges() == (int)aux_size / 6);

        static_assert(sizeof(Edge) == sizeof(u64));
        u32* node_data = (u32*)g.nodes.begin();
        u32* edge_data = (u32*)g.edge_data.begin();

        radix_pass_lsb_last<0,8>(aux2, edge_data, aux_size, first_1, node_data);
       
        node_data[0] = 0;
        for (u32 i = 1; i < g.nodes.size(); ++i) {
            if (node_data[i] == 0) g.nodes[i] = g.nodes[i-1];
        }
 
        delete[] first_0;
        delete[] first_1;
    }

    // aux1 is now owned by the buffer
    delete[] aux2;
    delete[] last_0;
    delete[] last_1;
    delete[] last_2;
    delete[] last_3;

    {float f = (float)(std::clock() - start_t) / (float)CLOCKS_PER_SEC;
    u32 bytes = (u32)(graph_data.size() / f);

    u64 hash_val = XXH64(graph_data.data(), graph_data.size(), 0);
    
    jout << "Done. (" << nice_bytes(graph_data.size()) << ", ";
    jout << jup_printf("%.2f", f) << "s, ";
    jout << nice_bytes(bytes) << "/s)\nChecksum (xxHash64): ";
    jout << nice_hex(hash_val) << endl;}

    } {

    auto start_t = std::clock();
    jout << "Compressing graph... ";
    jout.flush();
    
    Buffer lz4_data;
    
    lz4_data.emplace_back<u32>((u32)graph_data.size());
    lz4_data.emplace_back<u32>();
    int space_needed = LZ4_compressBound(graph_data.size());
    assert(space_needed > 0 /* Graph is too big! */);
    lz4_data.reserve_space(space_needed);

    int lz4_size = LZ4_compress_default(
        graph_data.data(), lz4_data.end(), graph_data.size(), lz4_data.space()
    );
    assert(lz4_size > 0);
    lz4_data.get<u32>(4) = lz4_size;
    lz4_data.addsize(lz4_size);
    out->write(lz4_data.data(), lz4_data.size());
    
    float f = (float)(std::clock() - start_t) / (float)CLOCKS_PER_SEC;
    u32 bytes = graph_data.size() / f;
    jout << "Done. (" << nice_bytes(lz4_size) << ", ";
    jout << jup_printf("%.2f", f) << "s, ";
    jout << nice_bytes(bytes) << "/s)\n" << endl;
    
    }
    
}

void graph_exec_jobfile(jup_str file, jup_str output) {
    Buffer jobfile;
    jobfile.read_from_file(file, false, 64*1024*1024);
    jobfile.append0();
    
    char* p = jobfile.begin();
    
    auto consume = [&jobfile, &p](jup_str str) {
        assert(jobfile.inside(p + str.size()));
        jup_str str2 {p, str.size()};
        if (str != str2) {
            jdbg < "Got " < Repr{str2} < "\b, expected " < Repr{str} ,0;
            die();
        }
        p += str.size();
    };
    auto consume_space = [&p]() {
        while (*p == ' ') ++p;
    };
    auto consume_line = [&p]() {
        while (*p == ' ' or *p == '\n') ++p;
    };
    
    consume("alarm_jobfile_header");
    errno = 0; int num_repo = std::strtol(p, &p, 0); assert_errno(errno != ERANGE);
    errno = 0; int num_file = std::strtol(p, &p, 0); assert_errno(errno != ERANGE);
    consume_line();
    
    jout << "The jobfile has " << num_repo << " repositories, in " << num_file << " files." << endl;

    Array<jup_str> repos {num_repo, true};
    Array<jup_str> files {num_file, true};

    for (int i = 0; i < num_repo; ++i) {
        consume("repo");
        consume_space();
        char* repo = p;
        while (*p != '\n') ++p;
        repos.push_back({repo, (int)(p - repo)});
        *p++ = 0;
        consume_line();
    }
    
    for (int i = 0; i < num_file; ++i) {
        consume("file");
        consume_space();
        char* file = p;
        while (*p != '\n') ++p;
        files.push_back({file, (int)(p - file)});
        *p++ = 0;
        consume_line();
    }

    while (p < jobfile.end() and *p == '\0') ++p;
    assert(p == jobfile.end());

    std::sort(repos.begin(), repos.end());
    int repo_count = 0;

    std::ofstream out_stream {output.c_str(), std::ios::binary};
    if (not out_stream) {
        jerr << "Error: opening output file " << output.c_str() << " failed.\n";
        die();
    }

    out_stream.write(SCHAFFILE_MAGIC.data(), SCHAFFILE_MAGIC.size());
    
    for (auto file: files) {
        jout << "Opening file " << file << endl;
        auto stream = alarm_init(file);
        while (not alarm_eof(&stream)) {
            auto repo = alarm_repo(&stream);
            if (not repo.size()) break;

            bool requested = std::binary_search(repos.begin(), repos.end(), repo);
            if (requested) {
                jout << "Found repository " << repo.c_str() << endl;
            
                graph_generate_single(&stream, &out_stream);
                if (++repo_count == repos.size()) {
                    jout << "All repositories found." << endl;
                    break;
                }
            } else {
                jout << "Skipping repository " << repo.c_str() << endl;
            
                while (not alarm_parse_eof(&stream)) {            
                    alarm_progress(&stream);
                    auto const& objects = alarm_parse(&stream);
                    assert(objects.size());
                }
            }
        }
        alarm_close(&stream);
    }
    
    // Make sure they are allowed to deallocate
    repos.trap_alloc(false);
    files.trap_alloc(false);
}

void graph_print_stats(jup_str input) {
    std::ifstream in_stream {input.c_str(), std::ios::binary};

    char buf[8];
    in_stream.read(buf, 4);
    assert(in_stream and in_stream.gcount() == 4);
    assert(std::memcmp(buf, SCHAFFILE_MAGIC.data(), 4) == 0);

    Buffer lz4_data;
    Buffer graph_data;
    while (in_stream) {
        in_stream.read(buf, 8);
        if (in_stream.eof() and in_stream.gcount() == 0) break;
        assert(in_stream and in_stream.gcount() == 8);
        
        u32 graph_size = *(u32*)buf;
        u32 lz4_size = *((u32*)buf + 1);

        lz4_data.resize(lz4_size);
        in_stream.read(lz4_data.data(), lz4_data.size());
        assert(in_stream and in_stream.gcount() == lz4_data.size());

        graph_data.resize(graph_size);
        int n = LZ4_decompress_safe(
            lz4_data.data(), graph_data.data(), lz4_data.size(), graph_data.size()
        );
        assert(n > 0 and n == graph_data.size());

        Graph const& g = graph_data.get<Graph>();
        u64 hash_val = XXH64(graph_data.data(), graph_data.size(), 0);
        u64 max_edges =  g.num_nodes() * (g.num_nodes() - 1);
        float density = (float)g.num_edges() / (float)max_edges;
    
        jout << "Number of nodes: " << g.num_nodes() << "\nNumber of edges: " << g.num_edges()
             << "\nGraph density: " << jup_printf("%.2f", density) << "%\nChecksum (xxHash64): ";
        jout << nice_hex(hash_val) << "\nUncompressed size: " << graph_data.size() << " (";
        jout << nice_bytes(graph_data.size()) << ")\nCompressed size: " << lz4_data.size() << " (";
        jout << nice_bytes(lz4_data.size()) << ")\n" << endl;
    }
    
}

} /* end of namespace jup */
