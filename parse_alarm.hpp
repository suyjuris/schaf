#pragma once

#include "libs/sha1.hpp"

#include "buffer.hpp"
#include "flat_data.hpp"
#include "idmap.hpp"

namespace jup {

constexpr int ALARM_REPO_MAX = 256;
constexpr int ALARM_GIT_HEADER_MAX = 256;
constexpr int ALARM_GIT_COMMIT_MAX = 64;
constexpr int ALARM_GIT_TREE_MAX = 4096 + 64;

// For efficiency we only use the first few bytes of the hashes. The collision
// probability is still acceptable.
using Sha_t = u64;

struct Git_tree;
struct Git_commit;

struct Git_object {
    enum Type: u8 {
        /* OBJ_BAD = -1, */
        OBJ_NONE = 0,
        OBJ_COMMIT = 1,
        OBJ_TREE = 2,
        OBJ_BLOB = 3,
        OBJ_TAG = 4,
        OBJ_OFS_DELTA = 6,
        OBJ_REF_DELTA = 7
    };
    Sha_t sha;
    u8 type;

    Git_commit* as_commit() {
        return type == OBJ_COMMIT ? (Git_commit*) this : nullptr;
    }
    Git_tree* as_tree() {
        return type == OBJ_TREE ? (Git_tree*) this : nullptr;
    }
    Git_commit const* as_commit() const {
        return type == OBJ_COMMIT ? (Git_commit const*) this : nullptr;
    }
    Git_tree const* as_tree() const {
        return type == OBJ_TREE ? (Git_tree const*) this : nullptr;
    }
};

struct Git_tree_Entry {
    enum Mode: u8 {
        NONE = 0, DIR, BLOB, BLOB_GROUP, BLOB_EXE, SYMLINK, GITLINK
    };
    Sha_t sha;
    u32 name;
    u8 mode;

    bool operator< (Git_tree_Entry const& other) const {
        return name < other.name;
    }
};

struct Git_tree: public Git_object {
    Flat_array<Git_tree_Entry, u16, u16> entries;
};

struct Git_commit: public Git_object {
    Sha_t tree;
    Flat_array<Sha_t> parents;
};

struct Alarm_stream {
    constexpr static int BUFFER_SIZE_IN  =  64*1024;
    constexpr static int BUFFER_SIZE_OUT = 512*1024;
    
    enum State: u8 {
        INIT = 0, REPO, PARSE_INIT, PARSE_MID, CLOSED
    };
    enum Z_state: u8 {
        Z_NONE = 0, Z_MID, Z_EOF
    };
    
    gzFile in_fd = nullptr;
    Buffer in_data;
    int in_data_off = 0;

    int in_data_znext = 0;
    u8 in_data_zstate = 0;

    SHA1_CTX hash_sha1;
    u32 hash_adler32;

    Buffer out_data;
    Idmap strings;
    u8 state = 0;

    std::time_t  last_progress_t = 0;
    std::clock_t last_progress_c = 0;
    u64 num_bytes = 0;
    int num_commits = 0;
    int num_trees = 0;
};

Alarm_stream alarm_init(Buffer_view fname);
Buffer_view alarm_repo(Alarm_stream* stream);
Flat_list<Git_object, u32, u32> const& alarm_parse(Alarm_stream* stream);
void alarm_close(Alarm_stream* stream);

bool alarm_parse_eof(Alarm_stream* stream);
bool alarm_eof(Alarm_stream* stream);

void alarm_progress(Alarm_stream* stream, int frequency = 5);

void alarm_benchmark(jup_str from);

} /* end of namespace jup */
