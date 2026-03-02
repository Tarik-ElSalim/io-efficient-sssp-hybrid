# heap_numba.py
"""
Heap operations for a numba binary min-heap using three parallel arrays for heap representation:
hk: The sorting criterion of the heap. This is where the numbers that determine the order are stored (e.g. distances).
hn: The corresponding “object IDs” or nodes. This is the payload that you link to the key (e.g. node ID).
hs: Extra information for each heap entry. Here, the current number of edges on the current path for exactly this heap
"""

# Standard library
from typing import Tuple

# Third-party
from numba import njit
import numpy as np
from numpy.typing import NDArray


@njit(cache=True, fastmath=True, inline='always')
def sift_up3(hk: np.ndarray, hn: np.ndarray, hs: np.ndarray, i: int) -> None:
    """
    Move the element at index i in a binary min-heap upwards
    until the heap property is satisfied. Only the key in hk is compared,
    hn and hs are swapped alongside to keep tuples aligned.

    :param hk,hs,hn as mentioned at top
    :param i: starting index for the sift-up (entry that had been inserted recently)
    :returns: None (in-place)
    """
    # while i is not the root
    while i > 0:
        # get the parent node of i at position floor((i-1)/2)
        p = (i - 1) >> 1
        # if parent node is bigger than i we change them, if stop with break
        if hk[i] < hk[p]:
            hk[i], hk[p] = hk[p], hk[i]
            hn[i], hn[p] = hn[p], hn[i]
            hs[i], hs[p] = hs[p], hs[i]
            i = p
        else:
            break


@njit(cache=True, fastmath=True, inline='always')
def sift_down3(hk: np.ndarray, hn: np.ndarray, hs: np.ndarray, i: int, size: int) -> None:
    """
    Restore the binary min-heap property by sifting the element at index `i`
    downward within the active range [0, size). Only the key in hk is compared,
    hn and hs are swapped alongside to keep tuples aligned.

    :param hk,hs,hn as mentioned at top
    :param i: starting index for the sift-down (entry that had been changed)
    :param size: number of elements in the heap
    :returns: None (in-place)
    """
    # loop that only stops when break
    while True:
        # gets the left child of i in the bin heap - is at position 2*i+1
        posl = (i << 1) + 1
        # If l out of range then no child exists so break
        if posl >= size:
            break
        # get the right child of i at position left child +1
        posr = posl + 1
        # choose the smaller child, l if there is no r else min(l,r)
        m = posl if (posr >= size or hk[posl] <= hk[posr]) else posr
        # check if the smaller child of i is smaller than i itself, if yes then switch them in all 3 arrays
        if hk[m] < hk[i]:
            hk[i], hk[m] = hk[m], hk[i]
            hn[i], hn[m] = hn[m], hn[i]
            hs[i], hs[m] = hs[m], hs[i]
            i = m
        # if the child is not smaller the order is correct then we can break
        else:
            break


@njit(cache=True, fastmath=True, inline='always')
def heap_pop3(hk: np.ndarray, hn: np.ndarray, hs: np.ndarray, size: int) \
        -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], int]:
    """
    Remove and return the min element (root) from the binary min-heap.

    :param hk,hs,hn as mentioned at top
    :param size: current number of elements in the heap (ensure that size > 0; not guarded!)
    :returns: key: smallest key (root); node: node id at root; step: hop count stored with that entry; last: new size
    """
    # sets new number of heap entries after pop
    last = size - 1
    # gets the root entry information that will be popped
    key = hk[0]
    node = hn[0]
    step = hs[0]
    # if heap exists after the pop move the last element to the root and the reorder the heap with sift_down
    if last > 0:
        hk[0] = hk[last]
        hn[0] = hn[last]
        hs[0] = hs[last]
        sift_down3(hk, hn, hs, 0, last)
    # returns the information about the popped root, changes heap inplace
    return key, node, step, last


@njit(cache=True, fastmath=True, inline='always')
def grow(hk: np.ndarray, hn: np.ndarray, hs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Double heap capacity by allocating larger arrays and copying existing data.

    :param hk,hs,hn as mentioned at top
    :returns: (hk2, hn2, hs2) — new arrays with 2x capacity, same contents in [0:size)
    """
    # Double size, create 3 news heap arrays, insert all heap entries and returns the 3 arrays at the end
    new_cap = hk.size * 2
    hk2 = np.empty(new_cap, hk.dtype)
    hk2[:hk.size] = hk
    hn2 = np.empty(new_cap, hn.dtype)
    hn2[:hn.size] = hn
    hs2 = np.empty(new_cap, hs.dtype)
    hs2[:hs.size] = hs
    return hk2, hn2, hs2


@njit(cache=True, fastmath=True, inline='always')
def ensure_push(hk: np.ndarray, hn: np.ndarray, hs: np.ndarray, size: int, key: float, node: int,
                steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Insert a (key, node, steps) triple into the binary min-heap (hk,hn,hs),
    growing capacity if needed, then restore heap order (sift-up).
    :param hk,hs,hn as mentioned at top
    :param size: current number of elements in the heap
    :param key: priority to insert (distance to node)
    :param node: node id associated with this key
    :param steps: hop count (number of edges from source for this entry)

    :returns: (hk, hn, hs, new_size)
              - hk/hn/hs: possibly reallocated arrays if growth occurred
              - new_size: size + 1 (the heap now contains one more element)
    """
    # If the heap is not big enough we double its size with the grow function
    if size >= hk.size:
        hk, hn, hs = grow(hk, hn, hs)
    # We set i = size, so we insert at the right index
    i = size
    # We insert the given key,node and steps value
    hk[i] = key
    hn[i] = node
    hs[i] = steps
    # And we ensure with sift up that the heap order is correct
    sift_up3(hk, hn, hs, i)
    # Return the result heap and size + 1 because we just added a new heap entry
    return hk, hn, hs, size + 1
