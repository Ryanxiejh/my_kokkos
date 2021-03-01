//
// Created by Ryanxiejh on 2021/3/1.
//

#ifndef MY_KOKKOS_KOKKOS_SYCL_ATOMIC_HPP
#define MY_KOKKOS_KOKKOS_SYCL_ATOMIC_HPP

namespace Kokkos{
namespace SyclAtomic{

typedef enum {
    SyclLocalSpace,
    SyclGlobalSpace
}SyclAtomicSpace;

template <typename T1, typename T2>
KOKKOS_INLINE_FUNCTION T1 atomic_fetch_add(T1* const ptr, T2 val, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        return a_ref.fetch_add(val);
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        return a_ref.fetch_add(val);
    }
}

template <typename T1, typename T2>
KOKKOS_INLINE_FUNCTION T1 atomic_fetch_sub(T1* const ptr, T2 val, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        return a_ref.fetch_sub(val);
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        return a_ref.fetch_sub(val);
    }
}

template <typename T1, typename T2>
KOKKOS_INLINE_FUNCTION T1 atomic_fetch_and(T1* const ptr, T2 val, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        return a_ref.fetch_and(val);
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        return a_ref.fetch_and(val);
    }
}

template <typename T1, typename T2>
KOKKOS_INLINE_FUNCTION T1 atomic_fetch_or(T1* const ptr, T2 val, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        return a_ref.fetch_or(val);
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        return a_ref.fetch_or(val);
    }
}

template <typename T1, typename T2>
KOKKOS_INLINE_FUNCTION T1 atomic_fetch_xor(T1* const ptr, T2 val, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        return a_ref.fetch_xor(val);
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        return a_ref.fetch_xor(val);
    }
}

template <typename T1, typename T2>
KOKKOS_INLINE_FUNCTION T1 atomic_fetch_min(T1* const ptr, T2 val, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        return a_ref.fetch_min(val);
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        return a_ref.fetch_min(val);
    }
}

template <typename T1, typename T2>
KOKKOS_INLINE_FUNCTION T1 atomic_fetch_max(T1* const ptr, T2 val, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        return a_ref.fetch_max(val);
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        return a_ref.fetch_max(val);
    }
}

template <typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION bool atomic_compare_exchange_weak(T1* const ptr, T2& expected, T3 desired, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        return a_ref.compare_exchange_weak(expected,desired);
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        return a_ref.compare_exchange_weak(expected,desired);
    }
}

template <typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION bool atomic_compare_exchange_strong(T1* const ptr, T2& expected, T3 desired, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        return a_ref.compare_exchange_strong(expected,desired);
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        return a_ref.compare_exchange_strong(expected,desired);
    }
}

template <typename T1, typename T2>
KOKKOS_INLINE_FUNCTION void atomic_store(T1* const ptr, T2 val, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        a_ref.store(val);
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        a_ref.store(val);
    }
}

template <typename T1>
KOKKOS_INLINE_FUNCTION T1 atomic_load(T1* const ptr, SyclAtomicSpace target_space){
    if(target_space == SyclLocalSpace){
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::local_space> a_ref(*ptr);
        return a_ref.load();
    }
    else{
        sycl::ONEAPI::atomic_ref<T1, sycl::ONEAPI::memory_order::seq_cst, sycl::ONEAPI::memory_scope::system,
                                sycl::access::address_space::global_space> a_ref(*ptr);
        return a_ref.load();
    }
}

template <typename T1>
KOKKOS_INLINE_FUNCTION void atomic_increment(T1* const ptr, SyclAtomicSpace target_space){
    atomic_fetch_add(ptr,1,target_space);
}

template <typename T1>
KOKKOS_INLINE_FUNCTION void atomic_decrement(T1* const ptr, SyclAtomicSpace target_space){
    atomic_fetch_sub(ptr,1,target_space);
}


} //namespace SyclAtomic
} //namespace Kokkos

#endif //MY_KOKKOS_KOKKOS_SYCL_ATOMIC_HPP
