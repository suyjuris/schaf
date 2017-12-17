
#include "parse_alarm.hpp"
#include "debug.hpp"
#include "utilities.hpp"

namespace jup {

static void stream_min(Alarm_stream* stream, int amount) {
    assert(stream);
    
    if (stream->in_data.size() - stream->in_data_off < amount) {
        auto p = stream->in_data.begin() + stream->in_data_off;
        int size = stream->in_data.end() - p;
        std::memmove(stream->in_data.begin(), p, size);
        stream->in_data.resize(size);
        stream->in_data_znext -= stream->in_data_off;
        stream->in_data_off = 0; 
        
        // duplicate
        int n = gzread(stream->in_fd, stream->in_data.end(), stream->in_data.space());
        if (n == -1) {
            int err;
            char const* msg = gzerror(stream->in_fd, &err);
            die(msg, err);
        }
        stream->num_bytes += n;
        stream->in_data.addsize(n);
    }
}

static bool stream_match(Alarm_stream* stream, jup_str str) {
    assert(stream);

    Alarm_stream stream2 = *stream;
    
    if (std::strncmp(
        stream->in_data.begin() + stream->in_data_off,
        str.c_str(),
        str.size()
    ) == 0) {
        stream->in_data_off += str.size();
        return true;
    } else {
        return false;
    }
}

static void stream_pop(Alarm_stream* stream, jup_str str) {
    assert(stream);
    if (not stream_match(stream, str)) {
        jdbg < "Got " < Repr{{stream->in_data.begin() + stream->in_data_off, str.size()}} < ", expected " < Repr{str} ,0;
        die();
    }
}

Alarm_stream alarm_init(jup_str fname) {
    Alarm_stream stream;
    stream.in_fd = gzopen(fname.c_str(), "rb");
    if (not stream.in_fd) {
        die("?errno while trying to open file %s", fname);
    }
    assert(gzbuffer(stream.in_fd, 512*1024) == 0);
    
    stream.in_data.reserve(Alarm_stream::BUFFER_SIZE_IN);
    stream.in_data.trap_alloc(true);
    stream.in_data_off = 0;
    stream.in_data_znext = 0;
    stream.out_data.reserve(Alarm_stream::BUFFER_SIZE_OUT);
    stream.out_data.trap_alloc(true);
    stream.state = Alarm_stream::REPO;

    jdbg.strings = &stream.strings;

    stream.last_progress_t = std::time(nullptr);
    stream.last_progress_c = std::clock();
    
    // Alarmfile magic
    stream_min(&stream, 4);
    stream_pop(&stream, "0\x9e\xb9\x08");
    
    return stream;
}

Buffer_view alarm_repo(Alarm_stream* stream) {
    assert(stream and stream->state == Alarm_stream::REPO);

    stream->num_commits = 0;
    stream->num_trees   = 0;
    stream->strings.reset();
    
    while (not alarm_eof(stream)) {
        stream_min(stream, ALARM_REPO_MAX);
        stream_pop(stream, "REPO ");
        int size = 0;
        while (stream->in_data[stream->in_data_off + size] != '\0') ++size;
        Buffer_view repo {stream->in_data.begin() + stream->in_data_off, size};
        stream->in_data_off += size + 1;

        if (stream->in_data[stream->in_data_off] == 'P') {
            // There was a bug in alarm, causing it to write only the header on non-existent
            // repositories.
            stream->state = Alarm_stream::PARSE_INIT;
            return repo;
        }
    }
    return Buffer_view {nullptr, 0};
}

static void parse_header(Alarm_stream* stream, u8* typ, u32* size) {
    assert(stream and typ and size);

    char c = stream->in_data[stream->in_data_off++];
    int i = 0;
    *typ = (c >> 4) & 7;
    *size = c & 15;
    while (c & 128) {
        c = stream->in_data[stream->in_data_off++];
        *size |= (c & 127) << ((++i)*7 - 3);
    }
    assert(0 <= i and i <= (int)sizeof(*size));
}

static char const* get_type_str(u8 type) {
    if (type == Git_object::OBJ_COMMIT) {
        return "commit";
    } else if (type == Git_object::OBJ_TREE) {
        return "tree";
    } else {
        assert(false);
        return nullptr;
    }
}

static void hash_init(Alarm_stream* stream, u8 type, int size) {
    assert(stream);

    stream->hash_adler32 = adler32(0, nullptr, 0);
        
    SHA1Init(&stream->hash_sha1);
    int n = std::snprintf(
        stream->out_data.end(), stream->out_data.space(), "%s %d", get_type_str(type), size
    );
    assert_errno(n != -1);
    assert(n < stream->out_data.space());
    
    SHA1Update(&stream->hash_sha1, (u8 const*)stream->out_data.end(), n + 1);
}

static void hash_update(Alarm_stream* stream, int amount) {
    assert(stream);

    Buffer_view data {&stream->in_data[stream->in_data_off], amount};
    stream->hash_adler32 = adler32(stream->hash_adler32, (u8 const*)data.data(), data.size());
    SHA1Update(&stream->hash_sha1, (u8 const*)data.data(), data.size());
}

static void stream_zinit(Alarm_stream* stream) {
    assert(stream and stream->in_data_zstate == Alarm_stream::Z_NONE);
    stream_min(stream, 2);
    u8 cmf = stream->in_data[stream->in_data_off++];
    u8 flg = stream->in_data[stream->in_data_off++];
    assert((cmf & 0x0f) == 8);
    assert((cmf*256 + flg) % 31 == 0);
    assert((flg & 32) == 0);
    stream->in_data_znext = stream->in_data_off;
    stream->in_data_zstate = Alarm_stream::Z_MID;
}

static void stream_skip_and_hash(Alarm_stream* stream, int offset) {
    assert(stream and offset >= 0);

    while(offset > stream->in_data.size()) {
        hash_update(stream, stream->in_data.size() - stream->in_data_off);
        stream->in_data_znext -= stream->in_data.size();
        offset -= stream->in_data.size();
        stream->in_data.resize(0);
        
        // duplicate
        int n = gzread(stream->in_fd, stream->in_data.end(), stream->in_data.space());
        if (n == -1) {
            int err;
            char const* msg = gzerror(stream->in_fd, &err);
            die(msg, err);
        }
        stream->num_bytes += n;
        stream->in_data.addsize(n);
        stream->in_data_off = 0;
    }
    hash_update(stream, offset - stream->in_data_off);
    stream->in_data_off = offset;
}

static void stream_zmin(Alarm_stream* stream, int amount) {
    assert(stream);
    
    if (stream->in_data_znext - stream->in_data_off < amount
        and stream->in_data_zstate == Alarm_stream::Z_MID)
    {
        
        stream_min(stream, amount + 5);
        u8* p = (u8*)&stream->in_data[stream->in_data_znext];
        u8 zhdr = *p++;
        if (zhdr & 1) {
            stream->in_data_zstate = Alarm_stream::Z_EOF;
        }
        assert((zhdr & 0xfe) == 0);
        u16 zsize  = *p++; zsize  |= *p++ << 8;
        u16 zsize0 = *p++; zsize0 |= *p++ << 8;
        assert((zsize ^ zsize0) == 0xffff);
        
        p = (u8*)stream->in_data.begin() + stream->in_data_off;
        std::memmove(p + 5, p, stream->in_data_znext - stream->in_data_off);
        stream->in_data_off += 5;
        stream->in_data_znext += 5 + zsize;
    } else {
        stream_min(stream, amount);
    }
}


static bool stream_zeof(Alarm_stream* stream) {
    assert(stream and (stream->in_data_zstate == Alarm_stream::Z_MID
        or stream->in_data_zstate == Alarm_stream::Z_EOF));

    stream_zmin(stream, 1);
    return (stream->in_data_zstate == Alarm_stream::Z_EOF
        and stream->in_data_off == stream->in_data_znext);
}

static void stream_zclose(Alarm_stream* stream) {
    assert(stream and (stream->in_data_zstate == Alarm_stream::Z_MID
        or stream->in_data_zstate == Alarm_stream::Z_EOF));

    while (stream->in_data_zstate == Alarm_stream::Z_MID) {
        stream_skip_and_hash(stream, stream->in_data_znext);
        stream_zmin(stream, 1);
    }
    stream_skip_and_hash(stream, stream->in_data_znext);

    stream_min(stream, 4);
    u8* p = (u8*)&stream->in_data[stream->in_data_off];
    u32 adler = *p++;
    adler = (adler << 8) | *p++;
    adler = (adler << 8) | *p++;
    adler = (adler << 8) | *p++;
    assert(stream->hash_adler32 == adler);
    stream->in_data_off += 4;

    stream->in_data_zstate = Alarm_stream::Z_NONE;
}

static void stream_zpop(Alarm_stream* stream, Buffer_view str) {
    assert(stream);
    hash_update(stream, str.size());
    stream_pop(stream, str);
}

static bool stream_zmatch(Alarm_stream* stream, Buffer_view str) {
    assert(stream);
    
    if (std::strncmp(
        stream->in_data.begin() + stream->in_data_off,
        str.c_str(),
        str.size()
    ) == 0) {
        hash_update(stream, str.size());
        stream->in_data_off += str.size();
        return true;
    } else {
        return false;
    }
}

static Sha_t zparse_sha(Alarm_stream* stream) {
    assert(stream);

    stream_zmin(stream, 20);
    hash_update(stream, 20);
        
    Sha_t result = 0;
    for (int i = 0; i < (int)sizeof(Sha_t); ++i) {
        u8 c = stream->in_data[stream->in_data_off++];
        result = (result << 8) | c;
    }
    stream->in_data_off += 20 - sizeof(Sha_t);
    return result;
}

static Sha_t zparse_sha_hex(Alarm_stream* stream) {
    assert(stream);

    stream_zmin(stream, 40);
    hash_update(stream, 40);
        
    Sha_t result = 0;
    for (int i = 0; i < (int)sizeof(Sha_t)*2; ++i) {
        char c = stream->in_data[stream->in_data_off++];
        if ('0' <= c and c <= '9') {
            result = (result << 4) | (c - '0');
        } else if ('a' <= c and c <= 'f') {
            result = (result << 4) | (c - 'a' + 10);
        } else {
            assert(false);
        }
    }
    stream->in_data_off += 40 - 2*sizeof(Sha_t);
    return result;
}

bool alarm_parse_eof(Alarm_stream* stream) {
    assert(stream);
    switch (stream->state) {
    case Alarm_stream::PARSE_INIT:
    case Alarm_stream::PARSE_MID:
        return false;
    case Alarm_stream::REPO:
        return true;
    default:
        assert(false);
        return false;
    }
}

bool alarm_eof(Alarm_stream* stream) {
    assert(stream);
    
    if (stream->state == Alarm_stream::REPO) {
        stream_min(stream, 1);
        return stream->in_data_off == stream->in_data.size();
    } else {
        return false;
    }
}

void alarm_progress(Alarm_stream* stream, int frequency) {
    assert(stream);
    
    auto now_t = std::time(nullptr);
    if (std::difftime(now_t, stream->last_progress_t) < frequency) return;
    
    stream->last_progress_t = now_t;

    int n = stream->in_data.size() - stream->in_data_off;
    u64 bytes = stream->num_bytes - n;
    stream->num_bytes = n;

    auto now_c = std::clock();
    float mbit = (float)bytes / (float)(now_c - stream->last_progress_c)
        * CLOCKS_PER_SEC / 1024.f / 1024.f;
    stream->last_progress_c = now_c;

    jout << "Reading... (commits: " << stream->num_commits << ", trees: " << stream->num_trees
         << ", speed: " << jup_printf("%3.2f MiB/s", mbit) << ")" << endl;
}

Flat_list<Git_object, u32, u32> const& alarm_parse(Alarm_stream* stream) {
    assert(stream);
    stream->out_data.reset();
    stream->out_data.reserve(Alarm_stream::BUFFER_SIZE_OUT);
    
    auto* result = &stream->out_data.emplace<Flat_list<Git_object, u32, u32>>();
    result->init(&stream->out_data);

    if (stream->state == Alarm_stream::PARSE_INIT) {
        stream_min(stream, 12);
        stream_pop(stream, {"PACK\0\0\0\2", 8});
        // Just ignore the size; it is probably empty, anyways
        stream->in_data_off += 4;
        stream->state = Alarm_stream::PARSE_MID;
    }
    assert(stream->state == Alarm_stream::PARSE_MID);

    while (true) {
        stream_min(stream, ALARM_GIT_HEADER_MAX);
        int offset = stream->in_data_off;
        
        u8 type;
        u32 size;
        parse_header(stream, &type, &size);

        if (stream->out_data.space() < (int)size + 32) {
            if (result->size() == 0) {
                stream->out_data.reserve(size + 32);
                result = &stream->out_data.get<Flat_list<Git_object, u32, u32>>();
            } else {
                stream->in_data_off = offset;
                break;
            }
        }

        if (type == Git_object::OBJ_NONE) {
            stream_min(stream, 20);
            stream->in_data_off += 20;
            stream->state = Alarm_stream::REPO;
            break;
        } else if (type == Git_object::OBJ_COMMIT) {
            auto& commit = result->emplace_back<Git_commit>(&stream->out_data);
            commit.type = type;
            
            hash_init(stream, type, size);
            stream_zinit(stream);
            
            stream_zmin(stream, ALARM_GIT_COMMIT_MAX);
            stream_zpop(stream, "tree ");
            commit.tree = zparse_sha_hex(stream);
            stream_zpop(stream, "\n");
            
            commit.parents.init(&stream->out_data);
            while (true) {
                stream_zmin(stream, ALARM_GIT_COMMIT_MAX);
                if (not stream_zmatch(stream, "parent ")) break;
                commit.parents.push_back(zparse_sha_hex(stream), &stream->out_data);
                stream_zpop(stream, "\n");
            }
            stream_zclose(stream);
            
            u8 sha_buf[20];
            SHA1Final(sha_buf, &stream->hash_sha1);
            Sha_t sha = 0;
            for (int i = 0; i < (int)sizeof(Sha_t); ++i) {
                sha = (sha << 8) | sha_buf[i];
            }
            commit.sha = sha;

            ++stream->num_commits;
        } else if (type == Git_object::OBJ_TREE) {
            auto& tree = result->emplace_back<Git_tree>(&stream->out_data);
            tree.type = type;
            
            hash_init(stream, type, size);
            stream_zinit(stream);

            tree.entries.init(&stream->out_data);
            while (not stream_zeof(stream)) {
                Git_tree_Entry entry;
                stream_zmin(stream, ALARM_GIT_TREE_MAX);

                int off = stream->in_data_off;
                
                u32 mode = 0;
                while (true) {
                    u8 c = stream->in_data[off++];
                    if (c == ' ') break;
                    if (not ('0' <= c and c <= '7'))
                        jdbg < Repr{{stream->in_data.data() + stream->in_data_off - 10, 20}} < type < size,0;
                    assert('0' <= c and c <= '7');
                    mode = (mode << 3) | (c - '0');
                }
                switch (mode) {
                case  040000: entry.mode = Git_tree_Entry::DIR;        break;
                case 0100644: entry.mode = Git_tree_Entry::BLOB;       break;
                case 0100664: entry.mode = Git_tree_Entry::BLOB_GROUP; break;
                case 0100755: entry.mode = Git_tree_Entry::BLOB_EXE;   break;
                case 0120000: entry.mode = Git_tree_Entry::SYMLINK;    break;
                case 0160000: entry.mode = Git_tree_Entry::GITLINK;    break;
                    
                // Not standard, but observed in the wild
                case 0100777: entry.mode = Git_tree_Entry::BLOB_EXE;   break;
                    
                default:
                    jerr << "Warning: Unknown mode " << nice_oct(mode) << "\n";
                    entry.mode = Git_tree_Entry::BLOB; break;
                }

                int i = 0;
                while (stream->in_data[off + i] != '\0') ++i;
                Buffer_view name {&stream->in_data[off], i};
                entry.name = stream->strings.get_id_mod(name);
                off += i+1;
                
                hash_update(stream, off - stream->in_data_off);
                stream->in_data_off = off;

                entry.sha = zparse_sha(stream);

                tree.entries.push_back(entry, &stream->out_data);
            }
            stream_zclose(stream);
                        
            u8 sha_buf[20];
            SHA1Final(sha_buf, &stream->hash_sha1);
            Sha_t sha = 0;
            for (int i = 0; i < (int)sizeof(Sha_t); ++i) {
                sha = (sha << 8) | sha_buf[i];
            }
            tree.sha = sha;
            
            ++stream->num_trees;
        } else {
            jdbg < type < size ,0;
            assert(false);
        }
    }
    return *result;
}

void alarm_close(Alarm_stream* stream) {
    assert(stream);
    gzclose(stream->in_fd);
    stream->in_fd = nullptr;
    stream->state = Alarm_stream::CLOSED;

    stream->in_data .trap_alloc(false);
    stream->out_data.trap_alloc(false);

    jdbg.strings = nullptr;
}

void alarm_benchmark(jup_str from) {
    auto stream = alarm_init(from);
    while (not alarm_eof(&stream)) {
        auto repo = alarm_repo(&stream);
        if (not repo.size()) break;
        jout << "Found repository " << repo.c_str() << endl;

        while (not alarm_parse_eof(&stream)) {
            alarm_progress(&stream);
            
            auto const& objects = alarm_parse(&stream);
            assert(objects.size());
            for (auto const& i: objects) {
                if (i.type == Git_object::OBJ_COMMIT) {
                    // nothing
                } else if (i.type == Git_object::OBJ_TREE) {
                    // nothing
                } else {
                    assert(false);
                }
            }
        }
        jout << "Commits: " << stream.num_commits << ", Trees: "
             << stream.num_trees << endl;
    }
    alarm_close(&stream);
}

} /* end of namespace jup */
