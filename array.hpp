#pragma once

#include "buffer.hpp"

namespace jup {

/**
 * An array of values. THIS CLASS DOES NOT CALL DESTRUCTORS! Do not use it with
 * classes that care about their destructors being called. In particular, only
 * use it with PODs.
 */
template <typename _T>
class Array {
public:
    using T = _T;
    
	Array() {}
    explicit Array(int capacity, bool set_trap_alloc = false) {
        reserve(capacity);
        trap_alloc(set_trap_alloc);
    }
    explicit Array(Buffer&& buffer): m_data{buffer} {
        assert(m_data.size() % sizeof(T) == 0);
    }
    
	void reserve(int newcap) { m_data.reserve(newcap * sizeof(T)); }
    auto reserve_guard(int incr) { return m_data.reserve_guard(incr * sizeof(T)); }

    void assign_zero(int count) {
        m_data.resize(count * sizeof(T));
        std::memset(m_data.begin(), 0, m_data.size());
    }
    
	void pop_front(int i) { m_data.pop_front(i * sizeof(T)); }

	void resize(int nsize) { m_data.resize(nsize * sizeof(T)); }
	void addsize(int incr) { m_data.addsize(incr * sizeof(T)); }

	void reset() { m_data.reset(); }

	void free() { m_data.free(); }

	int size() const { return m_data.size() / sizeof(T); }
	int capacity() const { return m_data.capacity() / sizeof(T); }

	bool trap_alloc() const { return m_data.trap_alloc(); }

	bool trap_alloc(bool value) { return m_data.trap_alloc(value); }

	int space() const { return m_data.space() / sizeof(T); }
	void reserve_space(int atleast) { reserve(size() + atleast); }

	template <typename... Args>
	T& emplace(int index = 0, Args&&... args) {
        return m_data.emplace<T, Args...>(index * sizeof(T), std::forward<Args>(args)...);
	}
	template <typename T, typename... Args>
	T& emplace_back(Args&&... args) {
		return emplace<T>(size(), std::forward<Args>(args)...);
	}

    void push_back(T const& obj) {
        m_data.append(Buffer_view::from_obj(obj));
    }

    T pop_back() {
        assert(size() > 0);
        T result = back();
        addsize(-1);
        return result;
    }

	T* begin() {return (T*)m_data.begin(); }
	T* end()   {return (T*)m_data.end(); }
	T* data()  {return begin();}
	T const* begin() const {return (T*)m_data.begin(); }
	T const* end()   const {return (T*)m_data.end(); }
	T const* data()  const {return begin();}

    T& front() { return (*this)[0]; }
    T const& front() const { return (*this)[0]; }
    T& back() { return (*this)[size() - 1]; }
    T const& back() const { return (*this)[size() - 1]; }
    
	T& operator[] (int pos) {
		assert(0 <= pos and pos < size());
		return data()[pos];
	}
	T operator[] (int pos) const {
		assert(0 <= pos and pos < size());
		return data()[pos];
	}

    operator bool() const {
        return size() != 0;
    }
    
    Buffer m_data;
};

template <typename T>
struct Array_view {
	constexpr Array_view(T const* data = nullptr, int size = 0):
		m_data{data}, m_size{size} {assert(size >= 0);}
    constexpr Array_view(std::nullptr_t): Array_view{} {}
	
	Array_view(Array<T> const& buf):
        m_data{buf.begin()}, m_size{buf.size()} {}
	
	constexpr Array_view(std::vector<T> const& vec):
		Buffer_view{vec.data(), (int)(vec.size())} {}
	
	constexpr int size() const { return m_size; }
	
	constexpr T const* begin() const { return m_data; }
	constexpr T const* end()   const { return m_data + m_size; }
	constexpr T const* data()  const { return begin(); }

    T const& front() const { return (*this)[0]; }
    T const& back() const { return (*this)[size() - 1]; }
    
	T const& operator[] (int pos) const {
		assert(0 <= pos and pos < size());
		return data()[pos];
	}

    Array_view<T> subview(int pos, int size_) const {
        assert(0 <= pos and pos + size_ <= size());
        return {data() + pos, size_};
    }

    Buffer_view as_bytes() const {
        return {data(), size() * sizeof(T)};
    }

	u32 get_hash() const {
        return as_bytes().get_hash();
	}

	bool operator== (Array_view<T> const& other) const {
        return as_bytes() == other.as_bytes();
	}
	bool operator!= (Array_view<T> const& buf) const { return !(*this == buf); }

    constexpr operator bool() const { return data() and size(); }
    
	T const* m_data;
	int m_size;
};

template <typename T>
struct Array_view_mut {
	Array_view_mut(T* data = nullptr, int size = 0):
		m_data{data}, m_size{size} {assert(size >= 0);}
    Array_view_mut(std::nullptr_t): Array_view_mut{} {}
	
	Array_view_mut(Array<T>& buf):
        m_data{buf.begin()}, m_size{buf.size()} {}
	
	Array_view_mut(std::vector<T>& vec):
		Buffer_view{vec.data(), (int)(vec.size())} {}
	
	int size() const { return m_size; }
	
	T* begin() { return m_data; }
	T* end()   { return m_data + m_size; }
    T* data()  { return begin(); }

    T& front() { return (*this)[0]; }
    T& back() { return (*this)[size() - 1]; }
    
	T& operator[] (int pos) {
		assert(0 <= pos and pos < size());
		return data()[pos];
	}

    Array_view_mut<T> subview(int pos, int size_) {
        assert(0 <= pos and pos + size_ <= size());
        return {data() + pos, size_};
    }
    
    Buffer_view as_bytes() const {
        return {data(), size() * sizeof(T)};
    }

	u32 get_hash() const {
        return as_bytes().get_hash();
	}

	bool operator== (Array_view<T> const& other) const {
        return as_bytes() == other.as_bytes();
	}
	bool operator== (Array_view_mut<T> const& other) const {
        return as_bytes() == other.as_bytes();
	}
	bool operator!= (Array_view<T>  const& buf) const { return !(*this == buf); }
	bool operator!= (Array_view_mut const& buf) const { return !(*this == buf); }

    constexpr operator bool() const { return data() and size(); }
    
	T* m_data;
	int m_size;
};

} /* end of namespace jup */
