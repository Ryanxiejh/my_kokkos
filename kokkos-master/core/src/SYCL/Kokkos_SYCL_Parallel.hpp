//
// Created by Ryanxiejh on 2021/2/22.
//

#ifndef KOKKOS_SYCL_PARALLEL_HPP_
#define KOKKOS_SYCL_PARALLEL_HPP_

namespace Kokkos{
namespace Impl{

template <class FunctorType, class ExecPolicy>
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
        cl::sycl::queue& q = *instance.m_queue;

        q.wait();

        q.submit([functor, policy](cl::sycl::handler& cgh) {
            cl::sycl::range<1> range(policy.end() - policy.begin());

            cgh.parallel_for(range, [=](cl::sycl::item<1> item) {
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

}   //namespace Impl
}   //namespace Kokkos

#endif //KOKKOS_SYCL_PARALLEL_HPP_
