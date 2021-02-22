//
// Created by Ryanxiejh on 2021/2/22.
//

#ifndef KOKKOS_SYCL_PARALLEL_HPP_
#define KOKKOS_SYCL_PARALLEL_HPP_

namespace Kokkos{
    namespace Impl{

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelFor Kokkos::SYCL with RangePolicy */
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::SYCL> {
public:
    using Policy = Kokkos::RangePolicy<Traits...>;

private:
    using Member       = typename Policy::member_type;
    using WorkTag      = typename Policy::work_tag;
    using LaunchBounds = typename Policy::launch_bounds;

    const FunctorType m_functor;
    const Policy m_policy;

private:
    ParallelFor()        = delete;
    ParallelFor& operator=(const ParallelFor&) = delete;

    template <typename Functor>
    static void sycl_direct_launch(const Policy& policy, const Functor& functor) {
        // Convenience references
        const Kokkos::SYCL& space = policy.space();
        Kokkos::Impl::SYCLInternal& instance =
                *space.impl_internal_space_instance();
        sycl::queue& q = *instance.m_queue;

        q.wait();

        q.submit([functor, policy](sycl::handler& cgh) {
            sycl::range<1> range(policy.end() - policy.begin());

            cgh.parallel_for(range, [=](sycl::item<1> item) {
                const typename Policy::index_type id =
                        static_cast<typename Policy::index_type>(item.get_linear_id()) +
                        policy.begin();
                if constexpr (std::is_same<WorkTag, void>::value)
                    functor(id);
                else
                    functor(WorkTag(), id);
            });
        });

        q.wait();
    }

    // Indirectly launch a functor by explicitly creating it in USM shared memory
    void sycl_indirect_launch() const {
        std::cout << "sycl_indirect_launch !!!" << std::endl;
        const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
        auto usm_functor_ptr = sycl::malloc_shared(sizeof(FunctorType),queue);
        new (usm_functor_ptr) FunctorType(m_functor);
        sycl_direct_launch(m_policy,std::reference_wrapper(*(static_cast<FunctorType*>(usm_functor_ptr))));
        sycl::free(usm_functor_ptr,queue);
    }

public:
    using functor_type = FunctorType;

    void execute() const {
        // if the functor is trivially copyable, we can launch it directly;
        // otherwise, we will launch it indirectly via explicitly creating
        // it in USM shared memory.
        if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
            sycl_direct_launch(m_policy, m_functor);
        else
            sycl_indirect_launch();
    }

    ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
            : m_functor(arg_functor), m_policy(arg_policy) {}
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
/* ParallelFor Kokkos::SYCL with MDRangePolicy */
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>, Kokkos::SYCL> {
private:
    typedef Kokkos::MDRangePolicy<Traits...> MDRangePolicy;
    typedef typename MDRangePolicy::impl_range_policy Policy;

    typedef typename MDRangePolicy::work_tag WorkTag;

    typedef typename Policy::WorkRange WorkRange;
    typedef typename Policy::member_type Member;

    typedef typename Kokkos::Impl::SyclIterateTile<
            MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>
            iterate_type;

    const FunctorType m_functor;
    const MDRangePolicy m_mdr_policy;
    const Policy m_policy;  // construct as RangePolicy( 0, num_tiles
    // ).set_chunk_size(1) in ctor

    template <typename Functor>
    /*static*/ void sycl_direct_launch(const Policy& policy, const Functor& functor) const{
        // Convenience references
        const Kokkos::SYCL& space = policy.space();
        Kokkos::Impl::SYCLInternal& instance = *space.impl_internal_space_instance();
        sycl::queue& q = *(instance.m_queue);

        q.wait();

        const typename Policy::index_type work_range = policy.end() - policy.begin();
        const typename Policy::index_type offset = policy.begin();

//    auto usm_functor_ptr = sycl::malloc_shared(sizeof(FunctorType),q);
//    new (usm_functor_ptr) FunctorType(m_functor);
//    FunctorType& func = std::reference_wrapper(*(static_cast<FunctorType*>(usm_functor_ptr)));

        MDRangePolicy mdr = m_mdr_policy;

        //std::cout << "work_range： " << work_range << std::endl;

        q.submit([=](sycl::handler& cgh) {
            sycl::range<1> range(work_range);
            sycl::stream out(1024, 256, cgh);
            cgh.parallel_for(range, [=](cl::sycl::item<1> item) {
                const typename Policy::index_type id =
                        static_cast<typename Policy::index_type>(item.get_linear_id()) + offset;
                const iterate_type iter(mdr,functor);
                iter(id);
                //functor(id);
//         if(id==0){
//             out << "sycl kernel run id: " << id << sycl::endl;
//             out << (iter.m_func.a)(id,0) << " " << (iter.m_func.a)(id,1) << " " << (iter.m_func.a)(id,2) << sycl::endl;
//         }
            });
        });

        q.wait();
    }

    //在usm中构造functor
    void sycl_indirect_launch() const {
        //这种方法在gpu上跑会有问题，所有的数据都变为0，而不是目标值
//    std::cout << "sycl_indirect_launch !!!" << std::endl;
//    const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
//    auto usm_functor_ptr = sycl::malloc_shared(sizeof(FunctorType),queue);
//    auto usm_iter_ptr = sycl::malloc_shared(sizeof(iterate_type),queue);
//    new (usm_functor_ptr) FunctorType(m_functor);
//    new (usm_iter_ptr) iterate_type(m_mdr_policy,*(static_cast<FunctorType*>(usm_functor_ptr)));
//    sycl_direct_launch(m_policy, std::reference_wrapper(*(static_cast<iterate_type*>(usm_iter_ptr))));
//    sycl::free(usm_functor_ptr,queue);
//    sycl::free(usm_iter_ptr,queue);

        std::cout << "sycl_indirect_launch !!!" << std::endl;
        const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
        auto usm_functor_ptr = sycl::malloc_shared(sizeof(FunctorType),queue);
        new (usm_functor_ptr) FunctorType(m_functor);
        sycl_direct_launch(m_policy,std::reference_wrapper(*(static_cast<FunctorType*>(usm_functor_ptr))));
        sycl::free(usm_functor_ptr,queue);
    }

public:
    void execute() const {
        if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
            sycl_direct_launch(m_policy, m_functor);
        else
            sycl_indirect_launch();
    }

    ParallelFor(const FunctorType &arg_functor, const MDRangePolicy &arg_policy)
            : m_functor(arg_functor),
              m_mdr_policy(arg_policy),
              m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)) {}
};
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelReduce with Kokkos::SYCL and RangePolicy */
template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType, Kokkos::SYCL> {
public:
    using Policy = Kokkos::RangePolicy<Traits...>;

private:
//   using Analysis =
//       FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;
//   using execution_space = typename Analysis::execution_space;
//   using value_type      = typename Analysis::value_type;
//   using pointer_type    = typename Analysis::pointer_type;
//   using reference_type  = typename Analysis::reference_type;
    typedef typename Policy::work_tag WorkTag;
    typedef Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
            FunctorType, ReducerType>
            ReducerConditional;
    typedef typename ReducerConditional::type ReducerTypeFwd;
    typedef
    typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
            WorkTag, void>::type WorkTagFwd;
    using ValueTraits =
    Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>;
    using value_type      = typename ValueTraits::value_type;
    using pointer_type    = typename ValueTraits::pointer_type;

//   using WorkTag = typename Policy::work_tag;
//   using ReducerConditional =
//       Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
//                          FunctorType, ReducerType>;
//   using WorkTagFwd =
//       std::conditional_t<std::is_same<InvalidType, ReducerType>::value, WorkTag,
//                          void>;
    using ValueInit =
    typename Kokkos::Impl::FunctorValueInit<FunctorType, WorkTagFwd>;

public:
    // V - View
    template <typename V>
    ParallelReduce(
            const FunctorType& f, const Policy& p, const V& v,
            typename std::enable_if<Kokkos::is_view<V>::value, void*>::type = nullptr)
            : m_functor(f), m_policy(p), m_result_ptr(v.data()) {}

    ParallelReduce(const FunctorType& f, const Policy& p,
                   const ReducerType& reducer)
            : m_functor(f),
              m_policy(p),
              m_reducer(reducer),
              m_result_ptr(reducer.view().data()) {}

//     template <typename T>
//     struct ExtendedReferenceWrapper : std::reference_wrapper<T> {
//         using std::reference_wrapper<T>::reference_wrapper;

//         //using value_type = typename FunctorValueTraits<T, WorkTag>::value_type;

//         template <typename Dummy = T>
//         std::enable_if_t<std::is_same_v<Dummy, T> &&
//                 ReduceFunctorHasInit<Dummy>::value>
//         init(value_type& old_value, const value_type& new_value) const {
//             return this->get().init(old_value, new_value);
//         }

//         template <typename Dummy = T>
//         std::enable_if_t<std::is_same_v<Dummy, T> &&
//                 ReduceFunctorHasJoin<Dummy>::value>
//         join(value_type& old_value, const value_type& new_value) const {
//             return this->get().join(old_value, new_value);
//         }

//         template <typename Dummy = T>
//         std::enable_if_t<std::is_same_v<Dummy, T> &&
//                 ReduceFunctorHasFinal<Dummy>::value>
//         final(value_type& old_value) const {
//             return this->get().final(old_value);
//         }
//     };

    template <typename PolicyType, typename Functor>
    void sycl_direct_launch(const PolicyType& policy,
                            const Functor& functor) const {
        // Convenience references
        const Kokkos::SYCL& space = policy.space();
        Kokkos::Impl::SYCLInternal& instance =
                *space.impl_internal_space_instance();
        cl::sycl::queue& q = *instance.m_queue;

        auto result_ptr = static_cast<pointer_type>(
                sycl::malloc(sizeof(*m_result_ptr), q, sycl::usm::alloc::shared));

        value_type identity{};
        if constexpr (!std::is_same<ReducerType, InvalidType>::value)
            m_reducer.init(identity);

        *result_ptr = identity;
        if constexpr (ReduceFunctorHasInit<Functor>::value)
            ValueInit::init(functor, result_ptr);

        q.submit([&](cl::sycl::handler& cgh) {
            // FIXME_SYCL a local size larger than 1 doesn't work for all cases
            cl::sycl::nd_range<1> range((policy.end() - policy.begin()), 100);

            const auto reduction = [&]() {
                if constexpr (!std::is_same<ReducerType, InvalidType>::value) {
                    return cl::sycl::ONEAPI::reduction(
                            result_ptr, identity,
                            [this](value_type& old_value, const value_type& new_value) {
                                m_reducer.join(old_value, new_value);
                                return old_value;
                            });
                } else {
                    if constexpr (ReduceFunctorHasJoin<Functor>::value) {
                        return cl::sycl::ONEAPI::reduction(
                                result_ptr, identity,
                                [functor](value_type& old_value, const value_type& new_value) {
                                    functor.join(old_value, new_value);
                                    return old_value;
                                });
                    } else {
                        return cl::sycl::ONEAPI::reduction(result_ptr, identity,
                                                           std::plus<>());
                    }
                }
            }();

            cgh.parallel_for(range, reduction,
                             [=](cl::sycl::nd_item<1> item, auto& sum) {
                                 const typename Policy::index_type id =
                                         static_cast<typename Policy::index_type>(
                                                 item.get_global_id(0)) +
                                         policy.begin();
                                 value_type partial = identity;
                                 if constexpr (std::is_same<WorkTag, void>::value)
                                     functor(id, partial);
                                 else
                                     functor(WorkTag(), id, partial);
                                 sum.combine(partial);
                             });
        });

        q.wait();

        static_assert(ReduceFunctorHasFinal<Functor>::value ==
                      ReduceFunctorHasFinal<FunctorType>::value);
        static_assert(ReduceFunctorHasJoin<Functor>::value ==
                      ReduceFunctorHasJoin<FunctorType>::value);

        if constexpr (ReduceFunctorHasFinal<Functor>::value)
            FunctorFinal<Functor, WorkTag>::final(functor, result_ptr);
        else
            *m_result_ptr = *result_ptr;

        sycl::free(result_ptr, q);
    }

    template <typename Functor>
    void sycl_indirect_launch(const Functor& functor) const {
//     // Convenience references
//     const Kokkos::SYCL& space = m_policy.space();
//     Kokkos::Impl::SYCLInternal& instance =
//         *space.impl_internal_space_instance();
//     Kokkos::Impl::SYCLInternal::IndirectKernelMemory& kernelMem =
//         *instance.m_indirectKernel;

//     // Allocate USM shared memory for the functor
//     kernelMem.resize(std::max(kernelMem.size(), sizeof(functor)));

//     // Placement new a copy of functor into USM shared memory
//     //
//     // Store it in a unique_ptr to call its destructor on scope exit
//     std::unique_ptr<Functor, Kokkos::Impl::destruct_delete> kernelFunctorPtr(
//         new (kernelMem.data()) Functor(functor));

        std::cout << "sycl_indirect_launch !!!" << std::endl;
        const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
        auto usm_functor_ptr = sycl::malloc_shared(sizeof(functor),queue);
        new (usm_functor_ptr) Functor(functor);
        //auto kernelFunctor = ExtendedReferenceWrapper<Functor>(*static_cast<Functor*>(usm_functor_ptr));
        auto kernelFunctor = std::reference_wrapper<Functor>(*static_cast<Functor*>(usm_functor_ptr));
        sycl_direct_launch(m_policy, kernelFunctor);
        sycl::free(usm_functor_ptr,queue);
    }

public:
    void execute() const {
        if (m_policy.begin() == m_policy.end()) {
            const Kokkos::SYCL& space = m_policy.space();
            Kokkos::Impl::SYCLInternal& instance =
                    *space.impl_internal_space_instance();
            cl::sycl::queue& q = *instance.m_queue;

            pointer_type result_ptr =
                    ReduceFunctorHasFinal<FunctorType>::value
                    ? static_cast<pointer_type>(sycl::malloc(
                            sizeof(*m_result_ptr), q, sycl::usm::alloc::shared))
                    : m_result_ptr;

            sycl::usm::alloc result_ptr_type =
                    sycl::get_pointer_type(result_ptr, q.get_context());

            switch (result_ptr_type) {
                case sycl::usm::alloc::host:
                case sycl::usm::alloc::shared:
                    ValueInit::init(m_functor, result_ptr);
                    break;
                case sycl::usm::alloc::device:
                    // non-USM-allocated memory
                case sycl::usm::alloc::unknown: {
                    value_type host_result;
                    ValueInit::init(m_functor, &host_result);
                    q.memcpy(result_ptr, &host_result, sizeof(host_result)).wait();
                    break;
                }
                default: Kokkos::abort("pointer type outside of SYCL specs.");
            }

            if constexpr (ReduceFunctorHasFinal<FunctorType>::value) {
                FunctorFinal<FunctorType, WorkTag>::final(m_functor, result_ptr);
                sycl::free(result_ptr, q);
            }

            return;
        }

        if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
            sycl_direct_launch(m_policy, m_functor);
        else
            sycl_indirect_launch(m_functor);
    }

private:
    FunctorType m_functor;
    Policy m_policy;
    ReducerType m_reducer;
    pointer_type m_result_ptr;
};

//----------------------------------------------------------------------------
/* ParallelReduce with Kokkos::SYCL and RangePolicy */
template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::MDRangePolicy<Traits...>, ReducerType, Kokkos::SYCL> {
public:
    using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
    using Policy = typename MDRangePolicy::impl_range_policy;

private:
    typedef typename Policy::work_tag WorkTag;
    typedef Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
            FunctorType, ReducerType>
            ReducerConditional;
    typedef typename ReducerConditional::type ReducerTypeFwd;
    typedef
    typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
            WorkTag, void>::type WorkTagFwd;
    using ValueTraits =
    Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>;
    using value_type      = typename ValueTraits::value_type;
    using pointer_type    = typename ValueTraits::pointer_type;
    using reference_type = typename ValueTraits::reference_type;

    using ValueInit =
    typename Kokkos::Impl::FunctorValueInit<FunctorType, WorkTagFwd>;

    typedef typename Kokkos::Impl::SyclIterateTile<
            MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, reference_type>
            iterate_type;
public:
    // V - View
    template <typename V>
    ParallelReduce(
            const FunctorType& f, const MDRangePolicy& p, const V& v,
            typename std::enable_if<Kokkos::is_view<V>::value, void*>::type = nullptr)
            : m_functor(f),
              m_mdr_policy(p),
              m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
              m_reducer(InvalidType()),
              m_result_ptr(v.data()) {}

    ParallelReduce(const FunctorType& f, const MDRangePolicy& p,
                   const ReducerType& reducer)
            : m_functor(f),
              m_mdr_policy(p),
              m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
              m_reducer(reducer),
              m_result_ptr(reducer.view().data()) {}

//     template <typename T>
//     struct ExtendedReferenceWrapper : std::reference_wrapper<T> {
//         using std::reference_wrapper<T>::reference_wrapper;

//         template <typename Dummy = T>
//         std::enable_if_t<std::is_same_v<Dummy, T> &&
//                 ReduceFunctorHasInit<Dummy>::value>
//         init(value_type& old_value, const value_type& new_value) const {
//             return this->get().init(old_value, new_value);
//         }

//         template <typename Dummy = T>
//         std::enable_if_t<std::is_same_v<Dummy, T> &&
//                 ReduceFunctorHasJoin<Dummy>::value>
//         join(value_type& old_value, const value_type& new_value) const {
//             return this->get().join(old_value, new_value);
//         }

//         template <typename Dummy = T>
//         std::enable_if_t<std::is_same_v<Dummy, T> &&
//                 ReduceFunctorHasFinal<Dummy>::value>
//         final(value_type& old_value) const {
//             return this->get().final(old_value);
//         }
//     };

    template <typename PolicyType, typename Functor>
    void sycl_direct_launch(const PolicyType& policy,
                            const Functor& functor) const {
        // Convenience references
        const Kokkos::SYCL& space = policy.space();
        Kokkos::Impl::SYCLInternal& instance =
                *space.impl_internal_space_instance();
        cl::sycl::queue& q = *instance.m_queue;

        auto result_ptr = static_cast<pointer_type>(
                sycl::malloc(sizeof(*m_result_ptr), q, sycl::usm::alloc::shared));

        value_type identity{};
        if constexpr (!std::is_same<ReducerType, InvalidType>::value)
            m_reducer.init(identity);

        *result_ptr = identity;
        if constexpr (ReduceFunctorHasInit<Functor>::value)
            ValueInit::init(functor, result_ptr);

        MDRangePolicy mdr = m_mdr_policy;

        q.submit([&](cl::sycl::handler& cgh) {
            // FIXME_SYCL a local size larger than 1 doesn't work for all cases
            cl::sycl::nd_range<1> range((policy.end() - policy.begin()), 100);

            const auto reduction = [&]() {
                if constexpr (!std::is_same<ReducerType, InvalidType>::value) {
                    return cl::sycl::ONEAPI::reduction(
                            result_ptr, identity,
                            [this](value_type& old_value, const value_type& new_value) {
                                m_reducer.join(old_value, new_value);
                                return old_value;
                            });
                } else {
                    if constexpr (ReduceFunctorHasJoin<Functor>::value) {
                        return cl::sycl::ONEAPI::reduction(
                                result_ptr, identity,
                                [functor](value_type& old_value, const value_type& new_value) {
                                    functor.join(old_value, new_value);
                                    return old_value;
                                });
                    } else {
                        return cl::sycl::ONEAPI::reduction(result_ptr, identity,
                                                           std::plus<>());
                    }
                }
            }();

            cgh.parallel_for(range, reduction,
                             [=](cl::sycl::nd_item<1> item, auto& sum) {
                                 const typename Policy::index_type id =
                                         static_cast<typename Policy::index_type>(
                                                 item.get_global_id(0)) +
                                         policy.begin();
                                 value_type partial = identity;
                                 const iterate_type iter(mdr,functor,partial);
                                 if constexpr (std::is_same<WorkTag, void>::value)
                                     //functor(id, partial);
                                     iter(id);
                                 else
                                     //functor(WorkTag(), id, partial);
                                     iter(WorkTag(),id);
                                 sum.combine(partial);
                             });
        });

        q.wait();

        static_assert(ReduceFunctorHasFinal<Functor>::value ==
                      ReduceFunctorHasFinal<FunctorType>::value);
        static_assert(ReduceFunctorHasJoin<Functor>::value ==
                      ReduceFunctorHasJoin<FunctorType>::value);

        if constexpr (ReduceFunctorHasFinal<Functor>::value)
            FunctorFinal<Functor, WorkTag>::final(functor, result_ptr);
        else
            *m_result_ptr = *result_ptr;

        sycl::free(result_ptr, q);
    }

    template <typename Functor>
    void sycl_indirect_launch(const Functor& functor) const {
        std::cout << "sycl_indirect_launch !!!" << std::endl;
        const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
        auto usm_functor_ptr = sycl::malloc_shared(sizeof(functor),queue);
        new (usm_functor_ptr) Functor(functor);
        auto kernelFunctor = std::reference_wrapper<Functor>(*static_cast<Functor*>(usm_functor_ptr));
        sycl_direct_launch(m_policy, kernelFunctor);
        sycl::free(usm_functor_ptr,queue);
    }

public:
    void execute() const {
        std::cout << "executing MDR parallel_reduce !!!" << std::endl;
        if (m_policy.begin() == m_policy.end()) {
            std::cout << "begin == end !!!" << std::endl;
            const Kokkos::SYCL& space = m_policy.space();
            Kokkos::Impl::SYCLInternal& instance =
                    *space.impl_internal_space_instance();
            cl::sycl::queue& q = *instance.m_queue;

            pointer_type result_ptr =
                    ReduceFunctorHasFinal<FunctorType>::value
                    ? static_cast<pointer_type>(sycl::malloc(
                            sizeof(*m_result_ptr), q, sycl::usm::alloc::shared))
                    : m_result_ptr;

            sycl::usm::alloc result_ptr_type =
                    sycl::get_pointer_type(result_ptr, q.get_context());

            switch (result_ptr_type) {
                case sycl::usm::alloc::host:
                case sycl::usm::alloc::shared:
                    ValueInit::init(m_functor, result_ptr);
                    break;
                case sycl::usm::alloc::device:
                    // non-USM-allocated memory
                case sycl::usm::alloc::unknown: {
                    value_type host_result;
                    ValueInit::init(m_functor, &host_result);
                    q.memcpy(result_ptr, &host_result, sizeof(host_result)).wait();
                    break;
                }
                default: Kokkos::abort("pointer type outside of SYCL specs.");
            }

            if constexpr (ReduceFunctorHasFinal<FunctorType>::value) {
                FunctorFinal<FunctorType, WorkTag>::final(m_functor, result_ptr);
                sycl::free(result_ptr, q);
            }

            return;
        }

        if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
            sycl_direct_launch(m_policy, m_functor);
        else{
            sycl_indirect_launch(m_functor);
            std::cout << "direct launch !!!" << std::endl;
        }
    }

private:
    FunctorType m_functor;
    MDRangePolicy m_mdr_policy;
    Policy m_policy;
    ReducerType m_reducer;
    pointer_type m_result_ptr;
};

}   //namespace Impl
}   //namespace Kokkos



#endif //KOKKOS_SYCL_PARALLEL_HPP_
