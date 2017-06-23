#pragma once

#include "array.hpp"
#include "buffer.hpp"

namespace jup {

class Arena_allocator {
public:
    Arena_allocator() = default;
    Arena_allocator(Arena_allocator const&) = default;
    Arena_allocator(Arena_allocator&&) = default;
    Arena_allocator& operator= (Arena_allocator const&) = default;
    Arena_allocator& operator= (Arena_allocator&&) = default;
    
    ~Arena_allocator() { free(); }

    void free() {
        for (auto i: pages) std::free((void*)i.data());
    }

    void store_view(Buffer_view page) {
        assert(page);
        if (pages.size()) {
            Buffer_view tmp = pages.back();
            pages.back() = page;
            pages.push_back(tmp);
        } else {
            last_page_used = page.size();
            pages.push_back(page);
        }
    }
    void store(Buffer* page) {
        assert(page);
        store_view(page->release());
    }
    
    void* allocate(int size) {
        if (size > page_size) {
            void* result = std::malloc(size);
            store_view(Buffer_view {result, size});
            return result;
        } else if (pages.size() and pages.back().size() - last_page_used <= size) {
            void* result = (void*)(pages.back().data() + last_page_used);
            last_page_used += size;
            return result;
        } else {
            void* result = std::malloc(page_size);
            pages.push_back({result, page_size});
            last_page_used = size;
            return result;
        }
    }
    
    template <typename T>
    bool inside(T const* ptr) {
        for (auto i: pages) {
            if (i.inside(ptr)) return true;
        }
        return false;
    }
    
    Array<Buffer_view> pages;
    int page_size = 64 * 1024;
    int last_page_used = 0;
};

} /* end of namespace jup */
