#pragma once

#include "array.hpp"
#include "buffer.hpp"

namespace jup {

class Arena {
public:
    Arena(int page_size = 4096): page_size{page_size} {}
    Arena(Arena const&) = default;
    Arena(Arena&&) = default;
    Arena& operator= (Arena const&) = default;
    Arena& operator= (Arena&&) = default;
    
    ~Arena() { free(); }

    void free() {
        for (auto i: pages) std::free((void*)i.data());
        pages.reset();
    }
    void reset() {
        int max_i = -1;
        int max_val = 0;
        for (int i = 0; i < pages.size(); ++i) {
            if (pages[i].size() > max_val) {
                max_i = i;
                max_val = pages[i].size();
            }
        }
        if (max_i != -1) {
            auto tmp = pages[max_i];
            pages[max_i] = {nullptr, 0};
            free();
            pages.push_back(tmp);
        }
        last_page_used = 0;
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
    int page_size;
    int last_page_used = 0;
};

} /* end of namespace jup */
