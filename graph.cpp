
#include <unordered_map>
#include <map>

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
        /*val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9ull;
        val = (val ^ (val >> 27)) * 0x94d049bb133111ebull;
        val = val ^ (val >> 31);
        return val;*/
    
        
        // XXHash, specialized for a single u64
        // see https://github.com/Cyan4973/xxHash
        u64 h64 = PRIME64_5 + 12;
        h64 ^= rotate_left<31>(val * PRIME64_2) * PRIME64_1;
        h64  = rotate_left<27>(h64) * PRIME64_1 + PRIME64_4;
        h64  = (h64 ^ (h64 >> 33)) * PRIME64_2;
        h64  = (h64 ^ (h64 >> 29)) * PRIME64_3;
        h64 ^= h64 >> 32;
        return h64;
        
    }
};

static void add_recursive(
    Git_tree_Entry const& entry,
    Path_hash_t hash_prefix,
    std::unordered_map<Sha_t, Git_tree const*> const& trees,
    Array<Path_hash_t>* changed,
    int arg
) {
    Path_hash_t hash = concatenate_path(hash_prefix, entry.name);
                    
    //jdbg < arg < hash ,0;
    if (entry.mode == Git_tree_Entry::DIR) {
        for (auto const& i: trees.at(entry.sha)->entries) 
            add_recursive(i, hash, trees, changed, 9);
    } else {
        changed->push_back(hash);
    }
}

static void calculate_diff(
    Path_hash_t hash_prefix,
    Git_tree const& tree1,
    Git_tree const& tree2,
    std::unordered_map<Sha_t, Git_tree const*> const& trees,
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
                    calculate_diff(hash, *trees.at(tree1_el.sha), *trees.at(tree2_el.sha), trees, changed);
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
    std::map<Sha_t, Git_commit const*> commits;
    std::unordered_map<Sha_t, Git_tree const*> trees;
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

    jout << "Parsing complete, starting with graph generation..." << endl;

    std::unordered_map<Path_hash_t, u32> nodes;
    std::unordered_map<Edge_t, u32, Hasher_Edge_t> edges;
    Array<Path_hash_t> changed;

    //jdbg < stream->strings.data.capacity() ,0;
    
    int count_commits = 0;
    int count_edges  = 0;

    auto start_t = std::time(nullptr);
    auto beg_c   = std::clock();
    auto beg_t   = start_t;
    
    for (auto const& it: commits) {
        if (it.second->parents.size() == 0) continue;
        
        auto now_t = std::time(nullptr);
        if (std::difftime(now_t, beg_t) >= 5) {
            beg_t = now_t;
            float f = (float)count_commits / (float)commits.size() * 100.f;

            auto now_c = std::clock();
            float elapsed = (float)(now_c - beg_c) / (float)CLOCKS_PER_SEC;
            float speed = (float)count_edges / elapsed;
            count_edges = 0;
            beg_c = now_c;
            
            jout << jup_printf("Generating graph... (%2.2f%%, %3.2f edges/s)", f, speed) << endl;
        }
        ++count_commits;

        changed.reset();
        
        Git_tree const& tree1 = *trees.at(it.second->tree);
        Git_tree const& tree2 = *trees.at(commits.at(it.second->parents[0])->tree);
        
        calculate_diff(0, tree1, tree2, trees, &changed);

        /*int n = changed.size();
        if (n > 100) {
            jdbg < n ,0;
            edges.reserve(n*(n-1) / 2);
            }*/

        for (int i = 0; i < changed.size(); ++i) {
            u32 node_i_outer = nodes.try_emplace(changed[i], nodes.size()).first->second;
            
            for (int j = 0; j < i; ++j) {
                auto node_i = node_i_outer;
                u32 node_j = nodes.try_emplace(changed[j], nodes.size()).first->second;
                if (node_i > node_j) std::swap(node_i, node_j);
                
                // TODO: Remove this after debugging, does not hold in case of a collision
                assert(node_i < node_j);

                edges[((u64)node_i << 32) | (u64)node_j] += 1;
                ++count_edges;
            }
        }
    }

    float f = std::difftime(std::time(nullptr), start_t);
    jout << jup_printf("Finished in %.0fs.\n", f);
    jout << "The graph has " << nodes.size() << " nodes and " << edges.size() << " edges." << endl;
}

void graph_exec_jobfile(jup_str file) {
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
