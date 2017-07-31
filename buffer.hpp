#pragma once

namespace jup {

// Forward declaration of Buffer, for the constructor of Buffer_view.
class Buffer;

/**
 * A read only objects referencing a continuous memory region of a certain size
 * with arbitrary contents. Supports iteration over the bytes. Has no ownership
 * of any kind. This is a simple pointer + size combination.
 *
 * If you want to store a c-style string in here, use the size of the string
 * without the terminating zero and call c_str() to extract the string, instead
 * of data().
 */
struct Buffer_view {
	constexpr Buffer_view(void const* data = nullptr, int size = 0):
		m_data{data}, m_size{size}
    {
        assert(data == nullptr ? size == 0 : size >= 0);
    }
    constexpr Buffer_view(std::nullptr_t): Buffer_view{} {}
	
	Buffer_view(Buffer const& buf);
	
	template<typename T>
	constexpr Buffer_view(std::vector<T> const& vec):
		Buffer_view{vec.data(), (int)(vec.size() * sizeof(T))} {}
	
	template<typename T>
	constexpr Buffer_view(std::basic_string<T> const& str):
		Buffer_view{str.data(), (int)(str.size() * sizeof(T))} {}
	
	constexpr Buffer_view(char const* str):
		Buffer_view{str, (int)std::strlen(str)} {}


	/**
	 * Construct from arbitrary object. This is not a constructor due to the
	 * obvious overloading problems.
	 */
	template<typename T>
	constexpr static Buffer_view from_obj(T const& obj) {
		return Buffer_view {&obj, sizeof(obj)};
	}

	constexpr int size() const { return m_size; }
	
	constexpr char const* begin() const { return (char const*)m_data; }
	constexpr char const* end()   const { return (char const*)m_data + m_size; }
	constexpr char const* data()  const { return begin(); }

    char front() const { return (*this)[0]; }
    char back()  const { return (*this)[size() - 1]; }

	/**
	 * Provide access to the bytes, with bounds checking.
	 */
	char operator[] (int pos) const {
		assert(0 <= pos and pos < size());
		return data()[pos];
	}

	/**
	 * Return a c-style string. Same as data, but asserts that the character
	 * just behind the last one is zero.
	 */
	char const* c_str() const {
		assert(*(data() + size()) == 0);
		return data();
	}

    /**
     * Return whether the pointer is inside the buffer
     */
    template <typename T>
    bool inside(T const* ptr) const {
        // duplicates Buffer::inside
        return (void const*)begin() <= (void const*)ptr
            and (void const*)(ptr + 1) <= (void const*)end();
    }
    
	/**
	 * Generate a simple hash of the contents of this Buffer_view. An empty
	 * buffer must have a hash of 0.
	 */
	u32 get_hash() const {
        // FNV-1a algorithm
		u32 result = 2166136261;
		for (char c: *this) {
			result = (result ^ c) * 16777619;
		}
        // Always return 0 when the buffer is empty
		return result ^ 2166136261;
	}

	/**
	 * Compare for byte-wise equality. 
	 */
	bool operator== (Buffer_view const& buf) const {
		if (size() != buf.size()) return false;
        return std::memcmp(data(), buf.data(), size()) == 0;
	}
	bool operator!= (Buffer_view const& buf) const { return !(*this == buf); }

    /**
     * Compare lexicographically by bytes
     */
    int compare(Buffer_view const& buf) const {
        auto cmp1 = std::memcmp(data(), buf.data(), std::min(size(), buf.size()));
        auto cmp2 = (size() > buf.size()) - (size() < buf.size());
        return cmp1 ? cmp1 : cmp2;
	}
	bool operator< (Buffer_view const& buf) const { return compare(buf) < 0; }
    

    /**
     * Return whether the buffer is valid and not empty.
     */
    constexpr operator bool() const {
        return data() and size();
    }
    
	void const* m_data;
	int m_size;
};

using jup_str = Buffer_view;

inline std::ostream& operator<< (std::ostream& s, Buffer_view buf) {
    s.write(buf.data(), buf.size());
    return s;
}

struct Buffer_guard {
    Buffer* buf;
    int size_target;
    bool trap_alloc;

    Buffer_guard(): buf{nullptr}, size_target{0}, trap_alloc{false} {}
    Buffer_guard(Buffer& buf, int size_incr);

    Buffer_guard(Buffer_guard&& g) {
        std::swap(buf, g.buf);
        std::swap(size_target, g.size_target);
        std::swap(trap_alloc, g.trap_alloc);
    }
    Buffer_guard& operator= (Buffer_guard&& g) {
        std::swap(buf, g.buf);
        std::swap(size_target, g.size_target);
        std::swap(trap_alloc, g.trap_alloc);
        return *this;
    }
    
    ~Buffer_guard();
};

/**
 * A handle for a continuous region of memory that can dynamically expand, if
 * necessary. This is like std::vector<char> in many regards. It supports both
 * move and copy semantics and has ownership of the managed memory (meaning that
 * the memory is free'd an destruction). There are no guarantees made for the
 * contents of uninitialized memory.
 *
 * There are three member variables:
 *  - the pointer to the memory, data()
 *  - the size of the allocated memory, capacity()
 *  - the amount of the memory that is used, size()
 * These are used together in the various methods. Of course, you may decide to
 * disregard size() completely and just use the block of memory.
 *
 * In debug mode (NDEBUG not defined) you can trap pointer invalidation due to
 * resizing.
 */
class Buffer {
#ifndef NDEBUG
	static_assert(sizeof(int) == 4, "Assuming 32bit ints for the bitmasks.");
#endif
public:

	/**
	 * These do what you would expect them to.
	 */
	Buffer() {}
    explicit Buffer(int capacity) {
        reserve(capacity);
    }
	Buffer(Buffer const& buf) { append(buf); }
	Buffer(Buffer&& buf) {
		m_data = buf.m_data;
		m_size = buf.m_size;
		m_capacity = buf.m_capacity;
		buf.m_data = nullptr;
		buf.m_size = 0;
		buf.m_capacity = 0;
	}

    ~Buffer() { free(); }

    Buffer& operator= (Buffer const& buf) {
		reset();
		append(buf);
		return *this;
	}
	Buffer& operator= (Buffer&& buf) {
		std::swap(m_data, buf.m_data);
		std::swap(m_size, buf.m_size);
		std::swap(m_capacity, buf.m_capacity);
		return *this;
	}

	/**
	 * Ensure that the Buffer has a capacity of at least newcap. If the current
	 * capacity is bigger, this does nothing. Else, new memory is allocated and
	 * the contents of the current block are moved. The new capacity is at least
	 * twice the old one.
	 */
	void reserve(int newcap) {
		if (capacity() < newcap) {
			assert(!trap_alloc());
			newcap = std::max(newcap, capacity() * 2);
			if (m_data) {
				m_data = (char*)std::realloc(m_data, newcap);
			} else {
				m_data = (char*)std::malloc(newcap);
			}
			// the trap_alloc flag is stored in m_capacity, don't disturb it
			m_capacity += newcap - capacity();
			assert(m_data);
		}
	}

    auto reserve_guard(int incr) {
        reserve_space(incr);
        return Buffer_guard {*this, incr};
    }
    
	/**
	 * Append the contents of the memory to this buffer.
	 */
	void append(void const* buf, int buf_size) {
		if (!buf_size) return;
		assert(buf_size > 0 and buf);
		if (capacity() < m_size + buf_size)
			reserve(m_size + buf_size);
		
		assert(capacity() >= m_size + buf_size);
		std::memcpy(m_data + m_size, buf, buf_size);
		m_size += buf_size;
	}
	void append(Buffer_view buffer) {
		append(buffer.data(), buffer.size());
	}
	void append0(int count = 1) {
		if (!count) return;
		assert(count > 0);
		if (capacity() < m_size + count)
			reserve(m_size + count);
		
		assert(capacity() >= count);
		std::memset(m_data + m_size, 0, count);
		m_size += count;
	}

	void pop_front(int i) {
		m_size -= i;
		std::memmove(m_data, m_data + i, m_size);
	}

	/**
	 * Change the size of the Buffer. Useful if you write to the memory
	 * manually.
	 */
	void resize(int nsize) {
		m_size = nsize;
		assert(m_size >= 0);
		reserve(m_size);
	}
	void addsize(int incr) {
		resize(m_size + incr);
	}

	/**
	 * Set the size to 0. Do not confuse this with free(), this does not
	 * actually release the memory.
	 */
	void reset() {
		m_size = 0;
	}

	/**
	 * Free the memory. Leaves the buffer in a valid state.
	 */
	void free() {
		assert(!trap_alloc());
		std::free(m_data);
		m_data = nullptr;
		m_size = 0;
		m_capacity = 0;
	}

    /**
     * Release ownership of the memory, and return a Buffer_view of the valid region.
     */
    Buffer_view release() {
        Buffer_view result {begin(), size()};
        m_data = nullptr;
        m_size = 0;
        m_capacity = 0;
        return result;
    }

    /**
     * Take ownership of the memory, free the current memory, if any.
     */
    void take(void* memory, int size) {
        free();
        m_data = (char*)memory;
        m_size = 0;
        m_capacity = size;
    }

	int size() const { return m_size; }
	int capacity() const {
		// If in debug mode, the most-significant bit of m_capacity is serving
		// as the trap_alloc flag.
#ifndef NDEBUG
		return m_capacity & 0x7fffffff;
#else
		return m_capacity;
#endif
	}

	/**
	 * Returns whether any pointer invalidation for pointers into the buffer may
	 * occur. If this is set, the program will abort on reallocation, which is
	 * useful for debugging.
	 */
	bool trap_alloc() const {
#ifndef NDEBUG
		return ((u32)m_capacity >> 31);
#else
		return false;
#endif
	}

	/**
	 * Maybe set the trap_alloc() flag and return its value.
	 */
	bool trap_alloc(bool value) {
#ifndef NDEBUG
		m_capacity ^= (u32)(trap_alloc() ^ value) << 31;
#endif					   
		return trap_alloc();
	}

	/**
	 * space() is the amount of space left in the Buffer (the capacity minus the
	 * size).
	 */
	int space() const {return capacity() - size();}
	/**
	 * Ensure, that atleast space is available.
	 */
	void reserve_space(int atleast) {
		reserve(size() + atleast);
	}

	/**
	 * A helper to save you some casting. Returns the memory offset by offset
	 * bytes interpreted as a T. Ensures that that much memory is
	 * available. This is of course not type-safe in any way, shape, or form.
	 */
	template <typename T>
	T& get(int offset = 0) {
		reserve(offset + sizeof(T));
		return *(T*)(m_data + offset);
	}
	template <typename T>
	T const& get_c(int offset = 0) const {
        assert(size() >= (int)(offset + sizeof(T)));
		return *(T const*)(m_data + offset);
	}
	/**
	 * Like get, but constructs the object in-place.
	 */
	template <typename T, typename... Args>
	T& emplace(int offset = 0, Args&&... args) {
		int end = offset + sizeof(T);
		reserve(end);
		if (m_size < end) resize(end);
		return *(new(m_data + offset) T {std::forward<Args>(args)...});
	}
	/**
	 * Like emplace, but contructs the object at the end.
	 */
	template <typename T, typename... Args>
	T& emplace_back(Args&&... args) {
		return emplace<T>(size(), std::forward<Args>(args)...);
	}

	char* begin() {return m_data;}
	char* end()   {return m_data + m_size;}
	char* data()  {return begin();}
	char const* begin() const {return m_data;}
	char const* end()   const {return m_data + m_size;}
	char const* data()  const {return begin();}

    char& front() { return (*this)[0]; }
    char& back()  { return (*this)[size() - 1]; }
    char  front() const { return (*this)[0]; }
    char  back()  const { return (*this)[size() - 1]; }

    /**
	 * Provide access to the buffer, with bounds checking.
	 */
	char& operator[] (int pos) {
		assert(0 <= pos and pos < size());
		return data()[pos];
	}
	char operator[] (int pos) const {
		assert(0 <= pos and pos < size());
		return data()[pos];
	}

	void write_to_file(Buffer_view filename, bool binary = true) {
		std::ofstream o;
        auto flags = (binary ? (std::ios::out | std::ios::binary) : std::ios::out);
		o.open(filename.c_str(), flags);
		o.write(data(), size());
		o.close();
	}

	void read_from_file(Buffer_view filename, bool binary = true, int maxsize = -1) {
		std::ifstream i;
        auto flags = (binary ? (std::ios::ate | std::ios::binary) : std::ios::ate);
		i.open(filename.c_str(), flags);
        std::streamsize fsize = i.tellg();
        assert(fsize <= (std::streamsize)maxsize);
        i.seekg(0, std::ios::beg);
        reserve_space(fsize);
        i.read(end(), fsize);
        assert(binary ? i.gcount() == fsize : i.gcount() <= fsize);
        addsize(i.gcount());
		i.close();
	}

    /**
     * Return whether the pointer is inside the buffer
     */
    template <typename T>
    bool inside(T const* ptr) const {
        // duplicates Buffer_view::inside
        return (void const*)begin() <= (void const*)ptr
            and (void const*)(ptr + 1) <= (void const*)end();
    }
    
	char* m_data = nullptr;
	int m_size = 0, m_capacity = 0;
};

inline Buffer_guard::Buffer_guard(Buffer& buf, int size_incr):
    buf{&buf}, size_target{buf.size() + size_incr}, trap_alloc{buf.trap_alloc()}
{
    buf.trap_alloc(true);
}
inline Buffer_guard::~Buffer_guard() {
    if (buf) {
        assert(buf->size() == size_target);
        buf->trap_alloc(trap_alloc);
    }
}

inline Buffer_view::Buffer_view(Buffer const& buf):
	Buffer_view{buf.data(), buf.size()} {}


} /* end of namespace jup */
