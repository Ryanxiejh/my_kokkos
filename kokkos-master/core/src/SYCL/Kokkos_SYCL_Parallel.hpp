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
            //sycl::stream out(1024, 256, cgh);
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
/* ParallelFor Kokkos::SYCL with TeamPolicy */
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Traits...>, Kokkos::SYCL> {
public:
    using Policy =
    Kokkos::Impl::TeamPolicyInternal<Kokkos::SYCL, Traits...>;
    using WorkTag = typename Policy::work_tag;
    using Member  = typename Policy::member_type;
    using size_type    = SYCL::size_type;

private:
    const FunctorType m_functor;
    const Policy m_policy;
    const size_type m_league_size;
    int m_team_size;
    const size_type m_vector_size;
    int m_shmem_size;
    int m_reduce_size;

public:

    template <typename Functor>
    void sycl_direct_launch(const Policy& policy, const Functor& functor) const {
        // Convenience references
        const Kokkos::SYCL& space = policy.space();
        Kokkos::Impl::SYCLInternal& instance =
                *space.impl_internal_space_instance();
        cl::sycl::queue& q = *instance.m_queue;

        q.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::nd_range<1> range(m_league_size * m_team_size, m_team_size);
            sycl::accessor<char, 1, sycl::access::mode::read_write, sycl::access::target::local>
                local_mem(m_shmem_size + m_reduce_size, cgh);
            const size_type league_size = m_league_size;
            int team_size = m_team_size;
            const size_type vector_size = m_vector_size;
            int shmem_size = m_shmem_size;
            int reduce_size = m_reduce_size;

            cgh.parallel_for(range, [=](cl::sycl::nd_item<1> item) {
                void* ptr = local_mem.get_pointer();
                int group_id = item.get_group_linear_id();
                int item_id = item.get_local_linear_id();
                Member member(ptr, reduce_size, (char*)ptr+reduce_size, shmem_size, group_id, league_size,
                              item_id, team_size, item);
                if constexpr (std::is_same<WorkTag, void>::value)
                    functor(member);
                else
                    functor(WorkTag(), member);

            });
        });

        q.wait();
    }

    //在usm中构造functor
    void sycl_indirect_launch() const {
        std::cout << "sycl_indirect_launch !!!" << std::endl;
        const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
        auto usm_functor_ptr = sycl::malloc_shared(sizeof(FunctorType),queue);
        new (usm_functor_ptr) FunctorType(m_functor);
        sycl_direct_launch(m_policy,std::reference_wrapper(*(static_cast<FunctorType*>(usm_functor_ptr))));
        sycl::free(usm_functor_ptr,queue);
    }

    void execute() const {
        if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
            sycl_direct_launch(m_policy, m_functor);
        else
            sycl_indirect_launch();
    }

    ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
        : m_functor(arg_functor),
          m_policy(arg_policy),
          m_league_size(arg_policy.league_size()),
          m_team_size(arg_policy.team_size()),
          m_vector_size(arg_policy.impl_vector_length()) {
        m_reduce_size = 100;
        m_shmem_size = (m_policy.scratch_size(0, m_team_size) + m_policy.scratch_size(1, m_team_size) +
                        FunctorTeamShmemSize<FunctorType>::value(m_functor, m_team_size));
        using namespace cl::sycl::info;
        if(m_reduce_size + m_shmem_size >
                arg_policy.space().impl_internal_space_instance()->m_queue->get_device().template get_info<device::local_mem_size>()){
            Kokkos::Impl::throw_runtime_exception(std::string(
                    "Kokkos::Impl::ParallelFor< SYCL > insufficient shared memory"));
        }

    }

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
/* ParallelReduce with Kokkos::SYCL and MDRPolicy */
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
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
/* ParallelReduce Kokkos::SYCL with TeamPolicy */
template <class FunctorType,  class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Traits...>, ReducerType, Kokkos::SYCL> {
public:
    using Policy = Kokkos::Impl::TeamPolicyInternal<Kokkos::SYCL, Traits...>;
    using WorkTag = typename Policy::work_tag;
    using Member  = typename Policy::member_type;
    using size_type    = SYCL::size_type;

    using ReducerConditional =
    Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
            FunctorType, ReducerType>;
    using ReducerTypeFwd = typename ReducerConditional::type;
    using WorkTagFwd =
    typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
            WorkTag, void>::type;

    using ValueTraits =
    Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>;
    using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
    using ValueJoin = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;

    using pointer_type   = typename ValueTraits::pointer_type;
    using reference_type = typename ValueTraits::reference_type;
    using value_type     = typename ValueTraits::value_type;

private:
    const FunctorType m_functor;
    const Policy m_policy;
    ReducerType m_reducer;
    pointer_type m_result_ptr;
    const size_type m_league_size;
    int m_team_size;
    const size_type m_vector_size;
    int m_shmem_size;
    int m_reduce_size;

public:

    template <typename Functor>
    void sycl_direct_launch(const Policy& policy, const Functor& functor) const {
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

        const auto reduction = [&]() {
            if constexpr (!std::is_same<ReducerType, InvalidType>::value) {
                printf("ReducerType: Built-in Type !!! \n");
//                return cl::sycl::ONEAPI::reduction(
//                        result_ptr, identity,
//                        [=](value_type& old_value, const value_type& new_value) {
//                            m_reducer.join(old_value, new_value);
//                            return old_value;
//                        });
                return cl::sycl::ONEAPI::reduction(result_ptr, identity,
                                                   std::minus<>());
            } else {
                if constexpr (ReduceFunctorHasJoin<Functor>::value) {
                    printf("ReducerType: Custome Type !!! \n");
                    return cl::sycl::ONEAPI::reduction(
                            result_ptr, identity,
                            [functor](value_type& old_value, const value_type& new_value) {
                                functor.join(old_value, new_value);
                                return old_value;
                            });
                } else {
                    printf("ReducerType: InvalidType(default) !!! \n");
                    return cl::sycl::ONEAPI::reduction(result_ptr, identity,
                                                       std::plus<>());
                }
            }
        }();

        q.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::nd_range<1> range(m_league_size * m_team_size, m_team_size);
            sycl::accessor<char, 1, sycl::access::mode::read_write, sycl::access::target::local>
                local_mem(m_shmem_size + m_reduce_size, cgh);
            const size_type league_size = m_league_size;
            int team_size = m_team_size;
            const size_type vector_size = m_vector_size;
            int shmem_size = m_shmem_size;
            int reduce_size = m_reduce_size;

            cgh.parallel_for(range, reduction,
                             [=](cl::sycl::nd_item<1> item, auto& sum) {
                void* ptr = local_mem.get_pointer();
                int group_id = item.get_group_linear_id();
                int item_id = item.get_local_linear_id();
                Member member(ptr, reduce_size, (char*)ptr+reduce_size, shmem_size, group_id, league_size,
                              item_id, team_size, item);
                value_type partial = identity;
                if constexpr (std::is_same<WorkTag, void>::value)
                    functor(member, partial);
                else
                    functor(WorkTag(), member, partial);
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

    //在usm中构造functor
    void sycl_indirect_launch() const {
        std::cout << "sycl_indirect_launch !!!" << std::endl;
        const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
        auto usm_functor_ptr = sycl::malloc_shared(sizeof(FunctorType),queue);
        new (usm_functor_ptr) FunctorType(m_functor);
        sycl_direct_launch(m_policy,std::reference_wrapper(*(static_cast<FunctorType*>(usm_functor_ptr))));
        sycl::free(usm_functor_ptr,queue);
    }

    void execute() const {
        if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
            sycl_direct_launch(m_policy, m_functor);
        else
            sycl_indirect_launch();
    }

    // V - View
    template <typename V>
    ParallelReduce(
            const FunctorType& arg_functor, const Policy& arg_policy, const V& v,
            typename std::enable_if<Kokkos::is_view<V>::value, void*>::type = nullptr)
            : m_functor(arg_functor),
              m_policy(arg_policy),
              m_reducer(InvalidType()),
              m_result_ptr(v.data()),
              m_league_size(arg_policy.league_size()),
              m_team_size(arg_policy.team_size()),
              m_vector_size(arg_policy.impl_vector_length()) {
        m_reduce_size = 100;
        m_shmem_size = (m_policy.scratch_size(0, m_team_size) + m_policy.scratch_size(1, m_team_size) +
                        FunctorTeamShmemSize<FunctorType>::value(m_functor, m_team_size));
        using namespace cl::sycl::info;
        if(m_reduce_size + m_shmem_size >
           arg_policy.space().impl_internal_space_instance()->m_queue->get_device().template get_info<device::local_mem_size>()){
            Kokkos::Impl::throw_runtime_exception(std::string(
                    "Kokkos::Impl::ParallelReduce< SYCL > insufficient shared memory"));
        }

    }

    ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                   const ReducerType& reducer)
            : m_functor(arg_functor),
              m_policy(arg_policy),
              m_reducer(reducer),
              m_result_ptr(reducer.view().data()),
              m_league_size(arg_policy.league_size()),
              m_team_size(arg_policy.team_size()),
              m_vector_size(arg_policy.impl_vector_length()) {
        m_reduce_size = 100;
        m_shmem_size = (m_policy.scratch_size(0, m_team_size) + m_policy.scratch_size(1, m_team_size) +
                        FunctorTeamShmemSize<FunctorType>::value(m_functor, m_team_size));
        using namespace cl::sycl::info;
        if(m_reduce_size + m_shmem_size >
           arg_policy.space().impl_internal_space_instance()->m_queue->get_device().template get_info<device::local_mem_size>()){
            Kokkos::Impl::throw_runtime_exception(std::string(
                    "Kokkos::Impl::ParallelReduce< SYCL > insufficient shared memory"));
        }

    }

};
//----------------------------------------------------------------------------



//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelScan with Kokkos::SYCL and RangePolicy */
template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::SYCL> {
private:
    using Policy      = Kokkos::RangePolicy<Traits...>;
    using WorkRange   = typename Policy::WorkRange;
    using WorkTag     = typename Policy::work_tag;
    using Member      = typename Policy::member_type;
    using ValueTraits = Kokkos::Impl::FunctorValueTraits<FunctorType, WorkTag>;
    using ValueInit   = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;
    using ValueJoin   = Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag>;
    using ValueOps    = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;

    using pointer_type   = typename ValueTraits::pointer_type;
    using value_type = typename ValueTraits::value_type;
    using reference_type = typename ValueTraits::reference_type;


public:
    ParallelScan(const FunctorType& f, const Policy& p)
            : m_functor(f),
              m_policy(p) {}

    template <typename PolicyType, typename Functor>
    void sycl_direct_launch(const PolicyType& policy,
                            const Functor& functor) const {
        // Convenience references
        const Kokkos::SYCL& space = policy.space();
        Kokkos::Impl::SYCLInternal& instance =
                *space.impl_internal_space_instance();
        cl::sycl::queue& q = *instance.m_queue;

        const std::size_t len = m_policy.end() - m_policy.begin();
        auto first_round_result = static_cast<pointer_type>(
                sycl::malloc(sizeof(value_type) * (len+1), q, sycl::usm::alloc::shared));

        q.submit([&](cl::sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                const typename Policy::index_type id =
                        static_cast<typename Policy::index_type>(item.get_id()) +
                        policy.begin();
                value_type update{};
                ValueInit::init(functor, &update);
                if constexpr (std::is_same<WorkTag, void>::value)
                    functor(id, update, false);
                else
                    functor(WorkTag(), id, update, false);
                ValueOps::copy(functor, &first_round_result[id-policy.begin()+1], &update); //初始化除[0]外的其他index的值
            });
        });

        q.wait();

        ValueInit::init(functor, &first_round_result[0]); //补充初始化[0]
        for(std::size_t i = 1; i < len+1 ; i++) {
            ValueJoin::join(functor, &first_round_result[i], &first_round_result[i - 1]);
        }
        std::cout << "first five value of first_round_result: "
                << first_round_result[0] << " "
                << first_round_result[1] << " "
                << first_round_result[2] << " "
                << first_round_result[3] << " "
                << first_round_result[4] << std::endl;

        q.submit([&](cl::sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                const typename Policy::index_type id =
                        static_cast<typename Policy::index_type>(item.get_id()) +
                        policy.begin();
//                value_type update{};
//                ValueInit::init(functor, &update);
                if constexpr (std::is_same<WorkTag, void>::value)
                    functor(id, first_round_result[id-policy.begin()], true);
                else
                    functor(WorkTag(), id, first_round_result[id-policy.begin()], true);
                //ValueOps::copy(functor, &first_round_result[id-policy.begin()], &update);
            });
        });

        q.wait();


        sycl::free(first_round_result, q);
    }

    template <typename Functor>
    void sycl_indirect_launch(const Functor& functor) const {
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

        if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
            sycl_direct_launch(m_policy, m_functor);
        else
            sycl_indirect_launch(m_functor);
    }

private:
    FunctorType m_functor;
    Policy m_policy;
};

template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>, ReturnType, Kokkos::SYCL> {
private:
    using Policy      = Kokkos::RangePolicy<Traits...>;
    using WorkRange   = typename Policy::WorkRange;
    using WorkTag     = typename Policy::work_tag;
    using Member      = typename Policy::member_type;
    using ValueTraits = Kokkos::Impl::FunctorValueTraits<FunctorType, WorkTag>;
    using ValueInit   = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;
    using ValueJoin   = Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag>;
    using ValueOps    = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;

    using pointer_type   = typename ValueTraits::pointer_type;
    using value_type = typename ValueTraits::value_type;
    using reference_type = typename ValueTraits::reference_type;


public:
    ParallelScanWithTotal(const FunctorType& f, const Policy& p, ReturnType& arg_returnvalue)
            : m_functor(f),
              m_policy(p),
              m_returnvalue(arg_returnvalue) {}

    template <typename PolicyType, typename Functor>
    void sycl_direct_launch(const PolicyType& policy,
                            const Functor& functor) const {
        // Convenience references
        const Kokkos::SYCL& space = policy.space();
        Kokkos::Impl::SYCLInternal& instance =
                *space.impl_internal_space_instance();
        cl::sycl::queue& q = *instance.m_queue;

        const std::size_t len = m_policy.end() - m_policy.begin();
        auto first_round_result = static_cast<pointer_type>(
                sycl::malloc(sizeof(value_type) * (len+1), q, sycl::usm::alloc::shared));

        q.submit([&](cl::sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                const typename Policy::index_type id =
                        static_cast<typename Policy::index_type>(item.get_id()) +
                        policy.begin();
                value_type update{};
                ValueInit::init(functor, &update);
                if constexpr (std::is_same<WorkTag, void>::value)
                    functor(id, update, false);
                else
                    functor(WorkTag(), id, update, false);
                ValueOps::copy(functor, &first_round_result[id-policy.begin()+1], &update);
            });
        });

        q.wait();

        ValueInit::init(functor, &first_round_result[0]);
        for(std::size_t i = 1; i < len+1 ; i++) {
            ValueJoin::join(functor, &first_round_result[i], &first_round_result[i - 1]);
        }
        std::cout << "first five value of first_round_result: "
                << first_round_result[0] << " "
                << first_round_result[1] << " "
                << first_round_result[2] << " "
                << first_round_result[3] << " "
                << first_round_result[4] << std::endl;

        q.submit([&](cl::sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
                const typename Policy::index_type id =
                        static_cast<typename Policy::index_type>(item.get_id()) +
                        policy.begin();
//                value_type update{};
//                ValueInit::init(functor, &update);
                if constexpr (std::is_same<WorkTag, void>::value)
                    functor(id, first_round_result[id-policy.begin()], true);
                else
                    functor(WorkTag(), id, first_round_result[id-policy.begin()], true);
                //ValueOps::copy(functor, &first_round_result[id-policy.begin()], &update);
            });
        });

        q.wait();

        const int size = ValueTraits::value_size(m_functor);
        DeepCopy<HostSpace, Kokkos::SYCLSpace, Kokkos::SYCL>(
                &m_returnvalue, &first_round_result[len-1], size);

        sycl::free(first_round_result, q);
    }

    template <typename Functor>
    void sycl_indirect_launch(const Functor& functor) const {
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

        if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
            sycl_direct_launch(m_policy, m_functor);
        else
            sycl_indirect_launch(m_functor);
    }

private:
    FunctorType m_functor;
    Policy m_policy;
    ReturnType& m_returnvalue;
};

//template <class FunctorType, class... Traits>
//class ParallelScanSYCLBase {
// public:
//  using Policy = Kokkos::RangePolicy<Traits...>;
//
// protected:
//  using Member       = typename Policy::member_type;
//  using WorkTag      = typename Policy::work_tag;
//  using WorkRange    = typename Policy::WorkRange;
//  using LaunchBounds = typename Policy::launch_bounds;
//
//  using ValueTraits = Kokkos::Impl::FunctorValueTraits<FunctorType, WorkTag>;
//  using ValueInit   = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;
//  using ValueJoin   = Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag>;
//  using ValueOps    = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;
//
// public:
//  using pointer_type   = typename ValueTraits::pointer_type;
//  using value_type     = typename ValueTraits::value_type;
//  using reference_type = typename ValueTraits::reference_type;
//  using functor_type   = FunctorType;
//  using size_type      = Kokkos::SYCL::size_type;
//  using index_type     = typename Policy::index_type;
//
// protected:
//  const FunctorType m_functor;
//  const Policy m_policy;
//  pointer_type m_scratch_space = nullptr;
//
// private:
//  template <typename Functor>
//  void scan_internal(cl::sycl::queue& q, const Functor& functor,
//                     pointer_type global_mem, std::size_t size) const {
//    // FIXME_SYCL optimize
//    constexpr size_t wgroup_size = 32;
//    auto n_wgroups               = (size + wgroup_size - 1) / wgroup_size;
//
//    // FIXME_SYCL The allocation should be handled by the execution space
//    auto deleter = [&q](value_type* ptr) { cl::sycl::free(ptr, q); };
//    std::unique_ptr<value_type[], decltype(deleter)> group_results_memory(
//        static_cast<pointer_type>(sycl::malloc(sizeof(value_type) * n_wgroups,
//                                               q, sycl::usm::alloc::shared)),
//        deleter);
//    auto group_results = group_results_memory.get();
//
//    q.submit([&](cl::sycl::handler& cgh) {
//      sycl::accessor<value_type, 1, sycl::access::mode::read_write,
//                     sycl::access::target::local>
//          local_mem(sycl::range<1>(wgroup_size), cgh);
//
//      // FIXME_SYCL we get wrong results without this, not sure why
//      sycl::stream out(1, 1, cgh);
//      cgh.parallel_for(
//          sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
//          [=](sycl::nd_item<1> item) {
//            const auto local_id  = item.get_local_linear_id();
//            const auto global_id = item.get_global_linear_id();
//
//            // Initialize local memory
//            if (global_id < size)
//              ValueOps::copy(functor, &local_mem[local_id],
//                             &global_mem[global_id]);
//            else
//              ValueInit::init(functor, &local_mem[local_id]);
//            item.barrier(sycl::access::fence_space::local_space);
//
//            // Perform workgroup reduction
//            for (size_t stride = 1; 2 * stride < wgroup_size + 1; stride *= 2) {
//              auto idx = 2 * stride * (local_id + 1) - 1;
//              if (idx < wgroup_size)
//                ValueJoin::join(functor, &local_mem[idx],
//                                &local_mem[idx - stride]);
//              item.barrier(sycl::access::fence_space::local_space);
//            }
//
//            if (local_id == 0) {
//              if (n_wgroups > 1)
//                ValueOps::copy(functor,
//                               &group_results[item.get_group_linear_id()],
//                               &local_mem[wgroup_size - 1]);
//              else
//                ValueInit::init(functor,
//                                &group_results[item.get_group_linear_id()]);
//              ValueInit::init(functor, &local_mem[wgroup_size - 1]);
//            }
//
//            // Add results to all items
//            for (size_t stride = wgroup_size / 2; stride > 0; stride /= 2) {
//              auto idx = 2 * stride * (local_id + 1) - 1;
//              if (idx < wgroup_size) {
//                value_type dummy;
//                ValueOps::copy(functor, &dummy, &local_mem[idx - stride]);
//                ValueOps::copy(functor, &local_mem[idx - stride],
//                               &local_mem[idx]);
//                ValueJoin::join(functor, &local_mem[idx], &dummy);
//              }
//              item.barrier(sycl::access::fence_space::local_space);
//            }
//
//            // Write results to global memory
//            if (global_id < size)
//              ValueOps::copy(functor, &global_mem[global_id],
//                             &local_mem[local_id]);
//          });
//    });
//
//    if (n_wgroups > 1) scan_internal(q, functor, group_results, n_wgroups);
//    q.wait();
//
//    q.submit([&](sycl::handler& cgh) {
//      cgh.parallel_for(sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
//                       [=](sycl::nd_item<1> item) {
//                         const auto global_id = item.get_global_linear_id();
//                         if (global_id < size)
//                           ValueJoin::join(
//                               functor, &global_mem[global_id],
//                               &group_results[item.get_group_linear_id()]);
//                       });
//    });
//    q.wait();
//  }
//
//  template <typename Functor>
//  void sycl_direct_launch(const Functor& functor) const {
//    // Convenience references
//    const Kokkos::SYCL& space = m_policy.space();
//    Kokkos::Impl::SYCLInternal& instance =
//        *space.impl_internal_space_instance();
//    cl::sycl::queue& q = *instance.m_queue;
//
//    const std::size_t len = m_policy.end() - m_policy.begin();
//
//    // Initialize global memory
//    q.submit([&](sycl::handler& cgh) {
//      auto global_mem = m_scratch_space;
//      auto policy     = m_policy;
//      cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
//        const typename Policy::index_type id =
//            static_cast<typename Policy::index_type>(item.get_id()) +
//            policy.begin();
//        value_type update{};
//        ValueInit::init(functor, &update);
//        if constexpr (std::is_same<WorkTag, void>::value)
//          functor(id, update, false);
//        else
//          functor(WorkTag(), id, update, false);
//        ValueOps::copy(functor, &global_mem[id], &update);
//      });
//    });
//    q.wait();
//
//    // Perform the actual exlcusive scan
//    scan_internal(q, functor, m_scratch_space, len);
//
//    // Write results to global memory
//    q.submit([&](sycl::handler& cgh) {
//      auto global_mem = m_scratch_space;
//      cgh.parallel_for(sycl::range<1>(len), [=](sycl::item<1> item) {
//        auto global_id = item.get_id();
//
//        value_type update = global_mem[global_id];
//        if constexpr (std::is_same<WorkTag, void>::value)
//          functor(global_id, update, true);
//        else
//          functor(WorkTag(), global_id, update, true);
//        ValueOps::copy(functor, &global_mem[global_id], &update);
//      });
//    });
//    q.wait();
//  }
//
//  template <typename Functor>
//  void sycl_indirect_launch(const Functor& functor) const {
//        std::cout << "sycl_indirect_launch !!!" << std::endl;
//        const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
//        auto usm_functor_ptr = sycl::malloc_shared(sizeof(functor),queue);
//        new (usm_functor_ptr) Functor(functor);
//        //auto kernelFunctor = ExtendedReferenceWrapper<Functor>(*static_cast<Functor*>(usm_functor_ptr));
//        auto kernelFunctor = std::reference_wrapper<Functor>(*static_cast<Functor*>(usm_functor_ptr));
//        sycl_direct_launch(kernelFunctor);
//        sycl::free(usm_functor_ptr,queue);
//  }
//
// public:
//  template <typename PostFunctor>
//  void impl_execute(const PostFunctor& post_functor) {
//    const auto& q = *(m_policy.space().impl_internal_space_instance()->m_queue);
//    const std::size_t len = m_policy.end() - m_policy.begin();
//
//    // FIXME_SYCL The allocation should be handled by the execution space
//    // consider only storing one value per block and recreate initial results in
//    // the end before doing the final pass
//    auto deleter = [&q](value_type* ptr) { cl::sycl::free(ptr, q); };
//    std::unique_ptr<value_type[], decltype(deleter)> result_memory(
//        static_cast<pointer_type>(sycl::malloc(sizeof(value_type) * len, q,
//                                               sycl::usm::alloc::shared)),
//        deleter);
//    m_scratch_space = result_memory.get();
//
//    if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
//      sycl_direct_launch(m_functor);
//    else
//      sycl_indirect_launch(m_functor);
//    post_functor();
//  }
//
//  ParallelScanSYCLBase(const FunctorType& arg_functor, const Policy& arg_policy)
//      : m_functor(arg_functor), m_policy(arg_policy) {}
//};
//
//template <class FunctorType, class... Traits>
//class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
//                   Kokkos::SYCL>
//    : private ParallelScanSYCLBase<FunctorType, Traits...> {
// public:
//  using Base = ParallelScanSYCLBase<FunctorType, Traits...>;
//
//  inline void execute() {
//    Base::impl_execute([]() {});
//  }
//
//  ParallelScan(const FunctorType& arg_functor,
//               const typename Base::Policy& arg_policy)
//      : Base(arg_functor, arg_policy) {}
//};
//
////----------------------------------------------------------------------------
//
//template <class FunctorType, class ReturnType, class... Traits>
//class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
//                            ReturnType, Kokkos::SYCL>
//    : private ParallelScanSYCLBase<FunctorType, Traits...> {
// public:
//  using Base = ParallelScanSYCLBase<FunctorType, Traits...>;
//
//  ReturnType& m_returnvalue;
//
//  inline void execute() {
//    Base::impl_execute([&]() {
//      const long long nwork = Base::m_policy.end() - Base::m_policy.begin();
//      if (nwork > 0) {
//        const int size = Base::ValueTraits::value_size(Base::m_functor);
//        DeepCopy<HostSpace, Kokkos::SYCLSpace>(
//            &m_returnvalue, Base::m_scratch_space + nwork - 1, size);
//      }
//    });
//  }
//
//  ParallelScanWithTotal(const FunctorType& arg_functor,
//                        const typename Base::Policy& arg_policy,
//                        ReturnType& arg_returnvalue)
//      : Base(arg_functor, arg_policy), m_returnvalue(arg_returnvalue) {}
//};

}   //namespace Impl
}   //namespace Kokkos




#endif //KOKKOS_SYCL_PARALLEL_HPP_
