
#include "libs/sparsehash.hpp"

#include "allocator.hpp"
#include "array.hpp"
#include "graph.hpp"
#include "debug.hpp"
#include "utilities.hpp"

namespace jup {

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

void _radix_sort_lsb(u64 *begin, u64 *end, u64 *begin1, u64 maxshift)
{
    u64 *end1 = begin1 + (end - begin);
     
    for (u64 shift = 0; shift <= maxshift; shift += 8)
        {
            size_t count[0x100] = {};
            for (u64 *p = begin; p != end; p++)
                count[(*p >> shift) & 0xFF]++;
            u64 *bucket[0x100], *q = begin1;
            for (int i = 0; i < 0x100; q += count[i++])
                bucket[i] = q;
            for (u64 *p = begin; p != end; p++)
                *bucket[(*p >> shift) & 0xFF]++ = *p;
            std::swap(begin, begin1);
            std::swap(end, end1);
        }
}
     
void _radix_sort_msb(u64 *begin, u64 *end, u64 *begin1, u64 shift)
{
    size_t count[0x100] = {};
    for (u64 *p = begin; p != end; p++)
        count[(*p >> shift) & 0xFF]++;
    u64 *bucket[0x100], *obucket[0x100], *q = begin1;
    for (int i = 0; i < 0x100; q += count[i++])
        obucket[i] = bucket[i] = q;
    for (u64 *p = begin; p != end; p++)
        *bucket[(*p >> shift) & 0xFF]++ = *p;
    for (int i = 0; i < 0x100; ++i)
        _radix_sort_lsb(obucket[i], bucket[i], begin + (obucket[i] - begin1), shift - 8);
}

void radix_sort(u64 *begin, u64 *end)
{
    assert(begin and end);
    u64 *begin1 = new u64[end - begin];
    _radix_sort_msb(begin, end, begin1, 56);
    delete[] begin1;
}

void _radix_sort_lsb_(u64* begin, u64* begin1, u64 size, u64 shift)
{
    u64* end = begin + size;
     
    u64 count[256] = {0};
    for (u64* p = begin; p != end; p++) {
        count[(*p >> shift) & 0xFF]++;
    }
    u64* bucket[256], *q = begin1;
    for (int i = 0; i < 256; ++i) {
        bucket[i] = q;
        q += count[i];
    }
    for (u64* p = begin; p != end; p++) {
        *bucket[(*p >> shift) & 0xff]++ = *p;
    }
}
     
void radix_sort_(u64* begin, u64* end)
{
    assert(begin and end);
    u64* begin1 = new u64[end - begin];

    u64 count[0x100] = {};
    for (u64* p = begin; p != end; p++) {
        count[(*p >> 40) & 0xff] += 1;
    }

    u64* bucket[256];
    u64* obucket[256];
    u64* q = begin1;
    for (int i = 0; i < 256; ++i) {
        obucket[i] = bucket[i] = q;
        q += count[i];
    }

    for (u64* p = begin; p != end; p++) {
        *bucket[(*p >> 40) & 0xff]++ = *p;
    }
    
    for (int i = 0; i < 0x100; ++i) {
        u64 size = bucket[i] - obucket[i];
        u64* obucket1_i = begin + (obucket[i] - begin1);
        _radix_sort_lsb_(obucket[i], obucket1_i, size,  0);
        _radix_sort_lsb_(obucket1_i, obucket[i], size,  8);
        _radix_sort_lsb_(obucket[i], obucket1_i, size, 32);
    }
    
    delete[] begin1;
}


void edge_sort_rec(u64* begin, u32 size, u32 nodes, u32* pointer_, u32* last) {
    u32* pointer = pointer_ + 1;
    std::memset(last, 0, nodes * sizeof(u32));
    
    for (u32 i = 0; i < size; ++i)
        ++last[(u32)(begin[i] & 0xffffffffull)];

    pointer[-1] = 0;
    pointer[0]  = 0;
    for (u32 i = 1; i < nodes; ++i) {
        pointer[i] =  last[i-1];
        last[i]    += last[i-1];
    }
    
    for (u32 i = 0; i < nodes; ++i) {
        while (pointer[i] != last[i]) {
            u64 value = begin[pointer[i]];
            u32 y = (u32)(value & 0xffffffffull);
            while (i != y) {
                u64 temp = begin[pointer[y]];
                begin[pointer[y]++] = value;
                value = temp;
                y = (u32)(value & 0xffffffffull);
            }
            begin[pointer[i]++] = value;
        }
    }
}

void edge_sort(u64* begin, u32 size, u32 nodes) {
    u32* pointer_ = new u32[nodes + 1];
    u32* pointer_2 = new u32[nodes + 1];
    u32* last = new u32[nodes] {0};
    u32* pointer = pointer_ + 1;
    
    for (u32 i = 0; i < size; ++i)
        ++last[(u32)(begin[i] >> 32)];

    pointer[-1] = 0;
    pointer[0]  = 0;
    for (u32 i = 1; i < nodes; ++i) {
        pointer[i] =  last[i-1];
        last[i]    += last[i-1];
    }
    
    for (u32 i = 0; i < nodes; ++i) {
        while (pointer[i] != last[i]) {
            u64 value = begin[pointer[i]];
            u32 y = (u32)(value >> 32);
            while (i != y) {
                u64 temp = begin[pointer[y]];
                begin[pointer[y]++] = value;
                value = temp;
                y = (u32)(value >> 32);
            }
            begin[pointer[i]++] = value;
        }
    }

    /*
    for (u32 i = 0; i < nodes; ++i) {
        int len = pointer_[i+1] - pointer_[i];
        //edge_sort_rec(begin + pointer_[i], len, nodes, pointer_2, last);
        }*/
    
    delete last;
    delete pointer_2;
    delete pointer_;
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

void graph_generate_single(Alarm_stream* stream) {
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
            
            jout << jup_printf("Generating graph... (%5.2f%%, %.0f edges/s)", f, speed) << endl;
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
    auto guard = graph_data.reserve_guard(Graph::total_space(nodes.size(), edges.size()));

    auto start_t = std::clock();

    Graph& g = graph_data.emplace_back<Graph>();

    g.nodes.init((u32)nodes.size() + 1, &graph_data);
    g.edge_data.init(edges.size() * 2, &graph_data);
    assert(g.num_nodes() == (int)nodes.size());
    assert(g.num_edges() == (int)edges.size());

    static_assert(sizeof(Edge) == sizeof(u64));
    Array_view_mut<u64> edge_data {(u64*)g.edge_data.begin(), (int)g.edge_data.size()};

    // The following algorithm is a bit tricky. First, we insert all edges (u, v) and (v, u)
    // into the buffer, for all adjacent nodes (u, v) with u < v. Then, the buffer is sorted,
    // which means that it contains all edges in the format
    //     (u1, u1_1), (u1, u1_2), ..., (u1, u1_n1), (u2, u2_1), ...
    // where u1 is the first node (with id 0), u1_1 is the first node adjacent to u1 (the one
    // with the smallest id), u1_u1 is the last node adjacent to u1, and so on. This we replace
    // with the following: (w(u, v) is the weight of edge (u, v))
    //     (u1_1, w(u1, u1_1)), (u1_2, w(u1, u1_2)), ...
    {int i = 0;
    for (auto it: edges) {
        u64 edge = it.first;
        edge_data[i++] = edge;
        edge_data[i++] = rotate_left<32>(edge);
    }
    assert(i == edge_data.size());}

    {
    auto start_t = std::clock();

    //std::sort(edge_data.begin(), edge_data.end());
    radix_sort_(edge_data.begin(), edge_data.end());
    //edge_sort(edge_data.begin(), edge_data.size(), nodes.size());
    
    float f = (float)(std::clock() - start_t) / (float)CLOCKS_PER_SEC;
    jout << jup_printf("%.2f", f) << "s" << endl;
    }

    g.nodes[0].data_offset = 0;
    for (int i = 0; i < edge_data.size(); ++i) {
        u64 edge = edge_data[i];
        u32 other = (u32)edge;
        u32 node = (u32)(edge >> 32);
        u32 weight = node > other ? edges[edge] : edges[rotate_left<32>(edge)];
        g.nodes[node+1].data_offset = i+1;
        g.edge_data[i].other  = other;
        g.edge_data[i].weight = weight;
    }
    for (u32 i = 1; i < g.nodes.size(); ++i) {
        if (g.nodes[i].data_offset == 0) g.nodes[i] = g.nodes[i-1];
    }

    float f = (float)(std::clock() - start_t) / (float)CLOCKS_PER_SEC;
    u32 bytes = (u32)(graph_data.size() / f);

    char hash[20];
    SHA1(hash, graph_data.data(), graph_data.size());
    char hash_str[8];
    for (u32 i = 0; i < sizeof(hash_str) / 2; i += 1) {
        char c1 = (u8)hash[i] >> 4;
        char c2 = (u8)hash[i] & 15;
        c1 = c1 < 10 ? c1 + '0' : c1 - 10 + 'a';
        c2 = c2 < 10 ? c2 + '0' : c2 - 10 + 'a';
        hash_str[2*i]     = c1;
        hash_str[2*i + 1] = c2;
    }
    hash_str[7] = 0;
    
    jout << "Finalized graph representation (" << nice_bytes(graph_data.size()) << ", ";
    jout << jup_printf("%.2f", f) << "s, ";
    jout << nice_bytes(bytes) << "/s), SHA1: " << hash_str << endl;

    //graph_data.write_to_file("graph.out");
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

    for (auto file: files) {
        jout << "Opening file " << file << endl;
        auto stream = alarm_init(file);
        while (not alarm_eof(&stream)) {
            auto repo = alarm_repo(&stream);
            if (not repo.size()) break;

            bool requested = std::binary_search(repos.begin(), repos.end(), repo);
            if (requested) {
                jout << "Found repository " << repo.c_str() << endl;
            
                graph_generate_single(&stream);
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

} /* end of namespace jup */
