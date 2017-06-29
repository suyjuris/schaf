
#pragma once

#include "buffer.hpp"
#include "flat_data.hpp"

namespace jup {
/**
 * A fast mapping from data to ids and back. This is mainly used to save space when using a small
 * amount of strings often, such as the tags in the OpenStreetMap database.
 *
 * It is implemented using an hashtable with open addressing. The hashtable maps the ids, which are
 * themselves equal to the hash of the data plus some small constant, to the offsets of the data
 * values.
 *
 * It consists only of two continuos memory blocks (the hashmap and the data); that makes it easy
 * to serialize and very space-efficient.
 * 
 * Template arguments:
 *   Offset_t  The datatype used for the offsets of the data values, this defines the maximum
 *		cumulative size of the data in bytes.
 *   Id_t  The datatype used for the ids, this is an upper bound on the number of different ids
 *   add_zero  Whether to automatically add a zero to the end of each data element; handy when
 *		using this structure for strings.
 */
template <typename _Offset_t = u32, typename _Id_t = u32, bool add_zero = true>
class Idmap_base {
	// The maximum load before enlarging.
    constexpr static float max_load = 0.75;
public:
    using Offset_t = _Offset_t;
    using Id_t = _Id_t;
    
	/**
	 * Initializes the map with 32 Slots for ids.
	 */ 
    Idmap_base(int capacity = 32) {
        data.reserve(256);
		// Need this so that no offset is zero or one
        data.append0(2);
        offsets.assign(capacity, 0);
    }

	/**
	 * Doubles the capacity of the hashmap.
	 */
    void enlarge() {
		// Sooo, it is not entirely clear that this does not invalidate ids. Let me explain:
		// The id is equal to the hash of the string plus maybe some small constant to keep it
		// unique: 
		//   id = hash + constant
		// Because an id points to the data, the following must hold before the enlarge (lets call
		// offsets.size() n, for simplicity):
		//   offsets[id % n] == offset_for_id
		// After the enlarge this must hold for 2*n instead of n, which is easy to ensure; if there
		// are no collisions due to the resizing:
		//      (hash1 + constant1) % n     != (hash2 + constant2) % n
		//   => (hash1 + constant1) % (2*n) != (hash2 + constant2) % (2*n)
		// It is quite easy to see that this does indeed hold, as any number divisable by 2*n is
		// also divisable by n. This is why we always double.
		// 
		// Additionally there must not be any 'holes' between hash % n and id % n, else get_id
		// would get stuck in them and not find the right id. For this to hold, is suffices to show
		// that all ids that had previously been incident continue to be after the enlarge:
		//      (hash1 + constant1 + 1) % n     == (hash2 + constant2) % n
		//   => (hash1 + constant1 + 1) % (2*n) == (hash2 + constant2) % (2*n)
		// This obviously does _not_ hold in general, which is why we will be inserting dummy
        // entries to fix this. (These are commonly called tombstones.) They do not negatively
		// impact the data storage density (as they may be overwritten), nor do they worsen the
		// lookup time (they only prevent it from improving due to the resize).

        std::vector<Offset_t> newoff;
        newoff.assign(offsets.size() * 2, 0);

        for (int index = 0; (size_t) index < offsets.size(); ++index) {
            Offset_t i = offsets[index];
            if (i == 0 or i == 1) continue;
			// Build the original id of this index; consisting of hash + constant
            Id_t hash = get_value(index).get_hash();
			size_t constant = ((index + offsets.size()) - hash % offsets.size()) % offsets.size();
            Id_t id = hash + constant;

			// Fill in the holes
			for (size_t i = 0; i < constant; ++i) {
				if (newoff[(hash + i) % newoff.size()] == 0)
					newoff[(hash + i) % newoff.size()] = 1;
			}
            newoff[id % newoff.size()] = i;
        }
        std::swap(offsets, newoff);
    }

	/**
	 * Returns the id of the data value obj if it is already inserted into the hashmap. Else, the
	 * behaviour is undefinded.
	 */
    Id_t get_id(Buffer_view obj) const {
        Id_t id = obj.get_hash();

        while ( offsets[id % offsets.size()] == 1
			|| (offsets[id % offsets.size()] != 0 && obj != get_value(id)))
			++id;
        assert(offsets[id % offsets.size()] > 1);
        return id;
    }

	/**
	 * Returns an id for the data value obj. It is inserted into the hashmap if needed.
	 */
    Id_t get_id_mod(Buffer_view obj) {
        Id_t hash = obj.get_hash();
		Id_t first_empty = offsets.size();

		// Try to find the value
		Id_t constant;
		for (constant = 0; constant < offsets.size(); ++constant) {
			if (offsets[(hash + constant) % offsets.size()] == 0) {
				break;
			} else if (offsets[(hash + constant) % offsets.size()] == 1) {
				if (first_empty == offsets.size())
					first_empty = constant;
			} else {
				if (obj == get_value(hash + constant)) {
					return hash + constant;
				}
			}
		}
		assert(constant < offsets.size());

		// We did not find the value, insert it
		if (first_empty == offsets.size()) {
			first_empty = constant;
		}
		Id_t id = hash + first_empty;
        offsets[id % offsets.size()] = data.size();
        data.append(Buffer_view::from_obj(obj.size()));
        data.append(obj);
        if (add_zero) {
            data.append({"", 1});
        }
        ++m_size;
        float load = (float) m_size / (float) offsets.size();
        if (load >= max_load) enlarge();

        return id;
    }

	/**
	 * Returns the data value for the id if the id has been inserted into the hashmap. Else, the
	 * behaviour is undefined. Like, really undefined. (No assertions and the fact that it returns ""
	 * is just you getting lucky!)
	 */
    Buffer_view get_value(Id_t id) const {
        return {data.data() + offsets[id % offsets.size()] + sizeof(int),
            *(int*) (data.data() + offsets[id % offsets.size()])};
    }

	/**
	 * Return the number of elements inside the hashmap.
	 */
    int size() const { return m_size; }

    void reset() {
        offsets.assign(offsets.size(), 0);
        data.resize(2);
    }

    void free() {
        offsets.assign(32, 0);
        offsets.shrink_to_fit();
        data.free();
        data.resize(2);
    }

	// Contains the data values
    Buffer data;
	
	// This is the hashmap. Each entry is one of:
	//  0:    meaning empty
	//  1:    also empty, but does not terminate search (i.e. may be between the expected position of
	//        an element and its actual position.) (Tombstone)
	//  else: the offset of the data value of this element
    std::vector<Offset_t> offsets;

	// The number of elements.
    int m_size = 0;
};

using Idmap = Idmap_base<>;

}
