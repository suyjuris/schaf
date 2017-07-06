#pragma once

#include "buffer.hpp"

namespace jup {

/**
 * This is a 'flat' data structure, meaning that it goes into a Buffer together
 * with its contents. It has the following layout:
 *               +--------------------------+
 *               |                          V
 *   ... first last ... next1 element1 ...  0 elementn ...
 *         |             ^ |                ^
 *         +-------------+ +------ ... -----+
 * The last-offset describes the difference between this and the address of the last next-offset
 * Iff start is 0, the list is empty
 * Iff the next-offset is 0, there are no further elements
 *
 * Else the object is invalid, and any method other than init() has undefined
 * behaviour.
 */

template <typename T, typename _Offset_t = u16>
struct Flat_list_iterator : public std::iterator<std::forward_iterator_tag, T> {
	using Type = T;
	using Offset_t = _Offset_t;

	char* ptr;

	Flat_list_iterator() { ptr = nullptr; }
	Flat_list_iterator(char* p) { ptr = p; }
	Flat_list_iterator(Flat_list_iterator const& orig) { ptr = orig.ptr; }
	bool operator==(Flat_list_iterator const& rhs) const { return ptr == rhs.ptr; }
	bool operator!=(Flat_list_iterator const& rhs) const { return ptr != rhs.ptr; }
	T& operator*() { assert(ptr); return *(T*)(ptr + sizeof(_Offset_t)); }
	T const& operator*() const { assert(ptr); return *(T const*)(ptr + sizeof(_Offset_t)); }
	Flat_list_iterator& operator++() {
		if (ptr == nullptr || *(Offset_t const*)ptr == 0) ptr = nullptr;
		else ptr += *(Offset_t const*)ptr;
		return *this;
	}
};
template <typename T, typename _Offset_t = u16, typename _Offset_big_t = _Offset_t>
struct Flat_list {
	using Type = T;
	using Offset_t = _Offset_t;
	using Offset_big_t = _Offset_big_t;
	using Iterator = Flat_list_iterator<T, _Offset_t>;

	Offset_big_t first;
	Offset_big_t last;

	Flat_list() : first{ 0 } {}
	Flat_list(Buffer* containing) { init(containing); }

	void init(Buffer* containing) {
		assert(containing);
		assert((void*)containing->begin() <= (void*)this
			and (void*)this < (void*)containing->end());
		assert(sizeof(Offset_big_t) >= sizeof(Offset_t));
		first = 0;
		last = 0;
	}

	int size() const {
		int size = 0;
		Offset_big_t next = first;
		char* ptr = (char*)this;
		while (next > 0) {
			ptr += next;
			++size;
			next = *(Offset_t*)ptr;
		}
		return size;
	}

	Iterator begin() { return Iterator(first ? (char*)this + first : nullptr); }
	Iterator end() { return Iterator(nullptr); }
	Iterator const begin() const { return Iterator(first ? (char*)this + first : nullptr); }
	Iterator const end()   const { return Iterator(nullptr); }

	T& front() { assert(first); return *(T*)((char*)this + first + sizeof(Offset_t)); }
	T& back() { assert(last); return *(T*)((char*)this + last + sizeof(Offset_t)); }
	T const& front() const { assert(first); return *(T const*)((char*)this + first + sizeof(Offset_t)); }
	T const& back() const { assert(last); return *(T const*)((char*)this + last + sizeof(Offset_t)); }

	/**
	* Return the element. Does bounds-checking.
	*/
	T& operator[] (int pos) {
		assert(pos >= 0);
		Offset_big_t next = first;
		char* ptr = (char*)this;
		while (pos --> 0) {
			assert(next);
			ptr += next;
			next = *(Offset_t*)ptr;
		}
		return *(T*)(ptr + sizeof(Offset_t));
	}
	T const& operator[] (int pos) const {
		assert(pos >= 0);
		Offset_big_t next = first;
		char* ptr = (char*)this;
		while (pos --> 0) {
			assert(next);
			ptr += next;
			next = *(Offset_t*)ptr;
		}
		return *(T const*)(ptr + sizeof(Offset_t));
	}

    void _advance(Buffer* containing) {
        assert(containing and containing->inside(this));
		if (first == 0) {
			narrow(first, containing->end() - (char*)this);
		} else {
			Offset_t* l = (Offset_t*)((char*)this + last);
            assert(*l == 0);
            narrow(*l, containing->end() - (char*)l);
		}
        narrow(last, containing->end() - (char*)this);
		containing->emplace_back<Offset_t>(Offset_t{0});
    }
    
	/**
     * Insert an element at the back. This operation may invalidate all pointers to
     * the Buffer, including the one you use for this object!
     */
    template <typename T2>
	T2& emplace_back(Buffer* containing) {
        _advance(containing);
		return containing->emplace_back<T2>();
	}
	void push_back(Buffer_view obj, Buffer* containing) {
        _advance(containing);
		containing->append(obj);
	}
    template <typename T2>
	void push_back(T2 const& obj, Buffer* containing) {
        push_back(Buffer_view::from_obj(obj), containing);
	}

	/**
	* Returns the smallest index containing an equal object or -1 if no such
	* object exists.
	*/
	int index(T const& obj) const {
		int i = 0;
		Offset_big_t next = first;
		char* ptr = (char*)this;
		while (next > 0) {
			if (*(ptr + sizeof(Offset_t)) == obj) return i;
			ptr += next;
			++i;
			next = *(Offset_t*)ptr;
		}
		return -1;
	}

	operator bool() const {
		return first;
	}

	/** 
	* Append another list being first element of list_buffer and its data. This operation may
	* invalidate all pointers to the Buffer, including the one you use for this object!
	*/
	void append(Buffer_view list_buffer, Buffer *containing) {
		Flat_list<T, Offset_t, Offset_big_t> const& list = *(Flat_list<T, Offset_t, Offset_big_t> const*)list_buffer.data();
		*(Offset_t*)((char*)this + last) = containing->end() + list.first - sizeof(Flat_list);
		last = containing->end() + list.last - sizeof(Flat_list);
		containing->append(list_buffer.data() + sizeof(Flat_list), list_buffer.size() - sizeof(Flat_list));
	}
};

/**
 * This is a 'flat' data structure, meaning that it goes into a Buffer together
 * with its contents. It has the following layout:
 * If start != 0:
 *   ... start ... size element1 element2 ...
 *         |        ^
 *         +--------+
 * Else the object is invalid, and any method other than init() has undefined
 * behaviour.
 *
 * Usage: Initialize this with the containing Buffer (using the contructor or
 * init()). Then, add the elements via push_back. Remember that each push_back
 * may invalidate all pointers into the Buffer including the pointer/reference
 * you use for the Flat_array! Circumvent this either by getting a new pointer
 * every time, or by reseving the memory that is needed (it then is also
 * recommended to set the trap_alloc() flag on the Buffer during
 * insertion). After initialization is finished, this object should be
 * considered read-only; new elements can only be appended at the end of the
 * Buffer (which has to be exactly the end of the Flat_array as well).
 */
template <typename T, typename _Offset_t = u16, typename _Size_t = u8>
struct Flat_array {
	using Type = T;
	using Offset_t = _Offset_t;
	using Size_t = _Size_t;
	
	Offset_t start;
		
	Flat_array(): start{0} {}
	Flat_array(Buffer* containing) { init(containing); }
    
    constexpr static int total_space(int size) {
        return sizeof(Flat_array<T, Offset_t, Size_t>) + extra_space(size);
    }
    constexpr static int extra_space(int size) {
        return sizeof(Size_t) + sizeof(T) * size;
    }

	/**
	 * Initializes the Flat_array by having it point to the end of the
	 * Buffer. The object must be contained in the Buffer!
	 */
	void init(Buffer* containing) {
		assert(containing and containing->inside(this));
		narrow(start, containing->end() - (char*)this);
        // This may invalidate us, but that is okay
		containing->emplace_back<Size_t>();
	}
	void init(int num_zeros, Buffer* containing) {
		assert(containing and containing->inside(this));
		narrow(start, containing->end() - (char*)this);
        // This may invalidate us, but that is okay
		containing->emplace_back<Size_t>((Size_t)num_zeros);
        containing->append0(num_zeros * sizeof(T));
	}

	void init(Flat_array<Type, Offset_t, Size_t> const& orig, Buffer* containing) {
		assert(containing and containing->inside(this) and not containing->inside(&orig));
		narrow(start, containing->end() - (char*)this);
        // This may invalidate us, but that is okay (note that orig is still valid)
		containing->append((char const*)&orig.m_size(), extra_space(orig.size()));
	}

    void init_safe(Flat_array<Type, Offset_t, Size_t> const& orig, Buffer* containing) {
        if (containing->inside(&orig)) {
            int offset_orig = (char const*)&orig.m_size() - containing->begin();
            int bytesize = extra_space(orig.size());
            narrow(start, containing->end() - (char*)this);
            
            // This may invalidate us, but that is okay
            auto g = containing->reserve_guard(bytesize);
            containing->append(containing->begin() + offset_orig, bytesize);
        } else {
            init(orig, containing);
        }
	}

	template<typename L_Offset_t, typename L_Offset_big_t>
	void init(Flat_list<T, L_Offset_t, L_Offset_big_t> const& orig, Buffer* containing) {
		assert(containing and containing->inside(this) and not containing->inside(&orig));
        assert(containing->space() >= extra_space(orig.size()));
		init(containing);
		L_Offset_big_t next = orig.first;
		char const* ptr = (char const*)&orig;
		while (next > 0) {
			ptr += next;
			next = *(L_Offset_t const*)ptr;
			push_back(*(T const*)(ptr + sizeof(L_Offset_t)), containing);
		}
	}

	Size_t size() const {
        if (!start) return 0;
        return m_size();
    }

	T* begin() { return start ? (T*)(&m_size() + 1) : nullptr; }
	T* end()   { return begin() + size(); }
	T const* begin() const { return start ? (T const*)(&m_size() + 1) : nullptr; }
	T const* end()   const { return begin() + size(); }

    T& front() {assert(size()); return *begin();}
    T& back()  {assert(size()); return end()[-1];}
    T const& front() const {assert(size()); return *begin();}
    T const& back()  const {assert(size()); return end()[-1];}

	/**
	 * Return the element. Does bounds-checking.
	 */
    T& operator[] (Size_t pos) {
		assert(0 <= pos and pos < size());
		return *(begin() + pos);
	}
	T const& operator[] (Size_t pos) const {
		assert(0 <= pos and pos < size());
		return *(begin() + pos);
	}


	/**
	 * Insert an element at the back. The end of the list and the end of the
	 * Buffer must be the same! This operation may invalidate all pointers to
	 * the Buffer, including the one you use for this object!
	 */
	void push_back(T const& obj, Buffer* containing) {
		assert(containing and containing->inside(this) and (void*)end() == (void*)containing->end());
		assert(++m_size() > 0);
		containing->emplace_back<T>(obj);
	}
	T& emplace_back(Buffer* containing) {
		assert(containing and containing->inside(this) and (void*)end() == (void*)containing->end());
		assert(++m_size() > 0);
		return containing->emplace_back<T>();
	}

	/**
	 * Count the number of objects equal to obj that are contained in this
	 * array.
	 */
	int count(T const& obj) const {
		int result = 0;
		for (auto& i: *this)
			if (i == obj) ++result;
		return result;
	}
    
	/**
	 * Returns the smallest index containing an equal object or -1 of no such
	 * object exists.
	 */
	int index(T const& obj) const {
        for (int i = 0; i < size(); ++i) {
            if ((*this)[i] == obj) return i;
        }
        return -1;
	}

	Size_t& m_size() {
		assert(start);
		return *(Size_t*)(((char*)this) + start);
	}
	Size_t const& m_size() const {
		assert(start);
		return *(Size_t const*)(((char*)this) + start);
	}

    operator bool() const {
        return start and m_size();
    }
};

/**
 * The same as Flat_array, but the offset is constant and the size part of the struct.
 *   ... size  ...  element1 element2 ...
 *         |          ^
 *         +--offset--+
 */
template <typename T, typename _Size_t, int offset>
struct Flat_array_const {
    static_assert(offset > 0);
	using Type = T;
	using Size_t = _Size_t;

    Size_t m_size;
		
	Flat_array_const(): m_size{0} {}
	Flat_array_const(Buffer* containing) { init(containing); }
    
	/**
	 * Initializes the Flat_array by having it point to the end of the
	 * Buffer. The object must be contained in the Buffer!
	 */
	void init(Buffer* containing) {
		assert(containing and containing->inside(this));
		assert(offset == containing->end() - (char*)this);
	}

	Size_t size() const { return m_size; }

	T* begin() { return (T*)((char*)this + offset); }
	T* end()   { return begin() + size(); }
	T const* begin() const { return (T*)((char*)this + offset); }
	T const* end()   const { return begin() + size(); }

    T& front() {assert(size()); return *begin();}
    T& back()  {assert(size()); return end()[-1];}
    T const& front() const {assert(size()); return *begin();}
    T const& back()  const {assert(size()); return end()[-1];}

	/**
	 * Return the element. Does bounds-checking.
	 */
	T& operator[] (Size_t pos) {
		assert(0 <= pos and pos < size());
		return *(begin() + pos);
	}
	T const& operator[] (Size_t pos) const {
		assert(0 <= pos and pos < size());
		return *(begin() + pos);
	}


	/**
	 * Insert an element at the back. The end of the list and the end of the
	 * Buffer must be the same! This operation may invalidate all pointers to
	 * the Buffer, including the one you use for this object!
	 */
	void push_back(T const& obj, Buffer* containing) {
		assert(containing and containing->inside(this) and (void*)end() == (void*)containing->end());
		assert(++m_size > 0); // be mindful of overflow
		containing->emplace_back<T>(obj);
	}
	T& emplace_back(Buffer* containing) {
		assert(containing and containing->inside(this) and (void*)end() == (void*)containing->end());
		assert(++m_size > 0); // be mindful of overflow
		return containing->emplace_back<T>();
	}

	/**
	 * Count the number of objects equal to obj that are contained in this
	 * array.
	 */
	int count(T const& obj) const {
		int result = 0;
		for (auto& i: *this)
			if (i == obj) ++result;
		return result;
	}
    
	/**
	 * Returns the smallest index containing an equal object or -1 of no such
	 * object exists.
	 */
	int index(T const& obj) const {
        for (int i = 0; i < size(); ++i) {
            if ((*this)[i] == obj) return i;
        }
        return -1;
	}

    operator bool() const { return m_size; }
};


template <typename _Offset_t = u16, typename _Size_t = u8>
struct Flat_array_ref_base {
    using Offset_t = _Offset_t;
    using Size_t = _Size_t;
    
    Offset_t offset = 0;
    u8 element_size = 0;

    template <typename T>
    Flat_array_ref_base(Flat_array<T, Offset_t, Size_t> const& arr, Buffer const& containing):
        element_size{sizeof(T)}
    {
        narrow(offset, (char*)&arr - containing.data()); 
        assert(containing.inside(&arr));
    }

    auto& ref(Buffer& container) const {
        assert(offset < container.size());
        return container.get<Flat_array<char, Offset_t, Size_t>>(offset);
    }
    
    Offset_t first_byte(Buffer& container) const {
        return offset;
    }
    Offset_t last_byte(Buffer& container) const {
        return offset + ref(container).start + sizeof(Size_t)
            + ref(container).size() * element_size;
    }

    bool operator== (Flat_array_ref_base<_Offset_t, _Size_t> const& other) const {
        return offset == other.offset and element_size == other.element_size;
    }
};

using Flat_array_ref = Flat_array_ref_base<>;

template <typename _Offset_t = u16, typename _Size_t = u8>
struct Diff_flat_arrays_base {
    struct Single_diff {
        u8 type;
        u8 arr;
        u8 size_index;
    };
    enum Type : u8 {
        ADD, REMOVE
    };

    Buffer* container;
    Buffer diffs;
    int _first;

    Diff_flat_arrays_base(Buffer* container): container{container} {
        assert(container);
        diffs.emplace_back<Flat_array<Flat_array_ref>>(&diffs);
    }

    template <typename Flat_array_t>
    void register_arr(Flat_array_t const& arr) {
        refs().push_back(Flat_array_ref {arr, *container}, &diffs);
        std::sort(refs().begin(), refs().end(), [this](Flat_array_ref a, Flat_array_ref b) {
            return a.first_byte(*container) < b.first_byte(*container);
        });
        _first = diffs.size();
    }

    int first() {
        assert(_first > 0);
        return _first;
    }
    void next(int* offset) {
        assert(offset);
        u8 type = diffs[*offset];
        if (type == ADD) {
            *offset += 2 + refs()[diffs[*offset+1]].element_size;
        } else if (type == REMOVE) {
            *offset += 3;
        } else {
            assert(false);
        }
        if (*offset >= diffs.size()) {
            assert(*offset == diffs.size());
            *offset = 0;
        }
    }

    template <typename Flat_array_t>
    void add(Flat_array_t const& arr, typename Flat_array_t::Type const& i) {
        int index = refs().index(Flat_array_ref {arr, *container});
        assert(index >= 0 /* array was not registered */);
        diffs.emplace_back<u8>(ADD);
        diffs.emplace_back<u8>((u8) index);
        assert(sizeof(i) == refs()[index].element_size);
        diffs.emplace_back<typename Flat_array_t::Type>(i);
    }
    
    template <typename Flat_array_t>
    void remove(Flat_array_t const& arr, int i) {
        int index = refs().index(Flat_array_ref {arr, *container});
        assert(index >= 0 /* array was not registered */);
        diffs.emplace_back<u8>(REMOVE);
        diffs.emplace_back<u8>((u8) index);
        diffs.emplace_back<u8>((u8) i);
    }

    Flat_array<Flat_array_ref>& refs() { return diffs.get<Flat_array<Flat_array_ref>>(); }

    void apply() {
        if (refs().size()) {
            assert(refs().back().last_byte(*container) == container->size());
        }
        for (int i = first(); i; next(&i)) {
            u8 type = diffs[i];
            u8 ref  = diffs[i+1];
            int adjust = refs()[ref].element_size * ((type == ADD) - (type == REMOVE));
            int remove_off = type == REMOVE ? - (refs()[ref].ref(*container).size()-1
                - diffs[i+2] ) * refs()[ref].element_size : 0;
            for (int j = 0; j < refs().size(); ++j) {
                if (j == ref) continue;
                if (refs()[j].first_byte(*container) < refs()[ref].last_byte(*container)
                    and refs()[j].last_byte(*container) > refs()[ref].last_byte(*container)) {
                    refs()[j].ref(*container).start += adjust;
                }
            }
            std::memmove(refs()[ref].ref(*container).end() + remove_off + adjust,
                         refs()[ref].ref(*container).end() + remove_off,
                         container->end() - (char*)refs()[ref].ref(*container).end() - remove_off);
            container->addsize(adjust);
            
            if (type == ADD) {
                std::memcpy(refs()[ref].ref(*container).end(),
                            diffs.data() + i+2, refs()[ref].element_size);
                refs()[ref].ref(*container).m_size() += 1;
            } else if (type == REMOVE) {
                refs()[ref].ref(*container).m_size() -= 1;
            } else {
                assert(false);
            }
        }
        diffs.resize(_first);
    }

    void reset() {
        diffs.reset();
        diffs.emplace_back<Flat_array<Flat_array_ref>>(&diffs);
        _first = 0;                                            
    }
};

using Diff_flat_arrays = Diff_flat_arrays_base<>;

/**
 * This is a 'flat' data structure, meaning that it goes into a Buffer together
 * with its contents. It has the following layout:
 *   ... map ... obj1 ... obj2 ...
 * where map is an array of offsets pointing to the strings. The obj need not be
 * continuous, the buffer can be used for other purposes in between.

 * This is a hashtable; it supports O(1) lookup and insertion (as long as it is
 * not too full). This hashtable specifically maps objects to integers. It is
 * guaranteed that an empty object is mapped to the id 0 if it is the first
 * object inserted. The inserted objects can be heterogeneous, but they are
 * compared by comparing their bytes.
 *
 * The template paramter add_zero causes a 0 to be appended after each block of
 * data containing an object. This is intended for the use with strings.
 */
template <typename Offset_t = u16, typename Id_t = u8, int Size = 256,
		  bool add_zero = true>
struct Flat_idmap_base {
	Offset_t map[Size];
	
	Flat_idmap_base(): map{0} {
		static_assert(Size <= std::numeric_limits<Id_t>::max() + 1,
					  "Id_t is not big enough for Size");
	}

	/**
	 * Return the id associated with the object obj. If it does not already
	 * exists the behaviour is undefined.
	 */
	Id_t get_id(Buffer_view obj) const {
		Id_t orig_id = obj.get_hash() % Size;
		Id_t id = orig_id;
		while (map[id] and obj != get_value(id)) {
			id = (id + 1) % Size;
			assert(id != orig_id);
		}
		assert(map[id]);
		return id;
	}

	/**
	 * Return the id associated with the object obj. If it does not already
	 * exists, it is inserted. The Buffer must contain the Flat_idmap object.
	 */
	Id_t get_id(Buffer_view obj, Buffer* containing) {
		assert(containing);
		assert((void*)containing->begin() <= (void*)this
			   and (void*)this < (void*)containing->end());
		
		Id_t orig_id = obj.get_hash() % Size;
		Id_t id = orig_id;
		while (map[id] and obj != get_value(id)) {
			id = (id + 1) % Size;
			assert(id != orig_id);
		}
		if (!map[id]) {
			map[id] = containing->end() - (char*)this;
			containing->append(Buffer_view::from_obj(obj.size()));
			containing->append(obj);
			if (add_zero) {
				containing->append({"", 1});
			}
		}
		return id;
	}

	/**
	 * Return the value of an id. If it does not already exists the behaviour is
	 * undefined.
	 */
	Buffer_view get_value(Id_t id) const {
		assert(0 <= id and id < Size and map[id]);
		return {(char*)this + map[id] + sizeof(int),
				*(int*)((char*)this + map[id])};
	}

};

using Flat_idmap = Flat_idmap_base<>;

/**
 * This is a 'flat' data structure, meaning that it goes into a Buffer together
 * with its contents. It has the following layout:
 * If start != 0:
 *   ... start ... obj ...
 *         |        ^
 *         +--------+
 * Else the object is invalid, and any method other than init() has undefined
 * behaviour.
 *
 * This models a reference, but is accessed like a pointer (e.g. using *, ->)
 */
template <typename T, typename _Offset_t = u16>
struct Flat_ref {
	using Type = T;
	using Offset_t = _Offset_t;

	Offset_t start;

	Flat_ref(): start{0} {}
	
	template <typename _T>
	Flat_ref(_T const& obj, Buffer* containing) {
		init(obj, containing);
	}

	/**
	 * Initializes the Flat_ref by having it point to the end of the Buffer. The
	 * object must be contained in the Buffer!
	 */
	template <typename _T>
	void init(_T const& obj, Buffer* containing) {
		assert(containing);
		assert((void*)containing->begin() <= (void*)this
			   and (void*)this < (void*)containing->end());
		narrow(start, containing->end() - (char*)this);
		containing->emplace_back<_T>(obj);
	}	

	T* ptr() {
		assert(start);
		return (T*)((char*)this + start);
	}
	T const* ptr() const {
		assert(start);
		return (T const*)((char*)this + start);
	}
	T* operator->() { return ptr(); }
	T const* operator->() const { return ptr(); }
	T& operator* () { return *ptr(); }
	T const& operator* () const { return *ptr(); }	
};

} /* end of namespace jup */

