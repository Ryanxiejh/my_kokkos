//
// Created by Ryanxiejh on 2021/2/24.
//

#ifndef MY_KOKKOS_KOKKOS_SYCL_TEAM_HPP
#define MY_KOKKOS_KOKKOS_SYCL_TEAM_HPP

namespace Kokkos{
namespace Impl{

//----------------------------------------------------------------------------
class SYCLTeamMember {
public:
    using execution_space      = Kokkos::SYCL;
    using scratch_memory_space = execution_space::scratch_memory_space;

private:
    mutable void* m_team_reduce;
    scratch_memory_space m_team_shared;
    int m_team_reduce_size;
    int m_league_rank;
    int m_league_size;
    int m_team_rank;
    int m_team_size;
    sycl::nd_item<1> m_item;

public:

    SYCLTeamMember(void* arg_team_reduce, const int arg_reduce_size, void* shared, const int shared_size,
                   const int arg_league_rank, const int arg_league_size,
                   const int arg_team_rank, const int arg_team_size, cl::sycl::nd_item<1> arg_item)
        : m_team_reduce(arg_team_reduce),
          m_team_shared(shared, shared_size),
          m_team_reduce_size(arg_reduce_size),
          m_league_rank(arg_league_rank),
          m_league_size(arg_league_size),
          m_team_rank(arg_team_rank),
          m_team_size(arg_team_size),
          m_item(arg_item) {}


    // Indices
    KOKKOS_INLINE_FUNCTION int league_rank() const { return m_league_rank; }
    KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size; }
    KOKKOS_INLINE_FUNCTION int team_rank() const { return m_team_rank; }
    KOKKOS_INLINE_FUNCTION int team_size() const { return m_team_size; }

    // Scratch Space
    KOKKOS_INLINE_FUNCTION
    const scratch_memory_space& team_shmem() const {
        return m_team_shared.set_team_thread_mode(0, 1, 0);
    }

    KOKKOS_INLINE_FUNCTION
    const scratch_memory_space& team_scratch(int level) const {
        return m_team_shared.set_team_thread_mode(level, 1, 0);
    }

    KOKKOS_INLINE_FUNCTION
    const scratch_memory_space& thread_scratch(int level) const {
        return m_team_shared.set_team_thread_mode(level, team_size(), team_rank());
    }

    // Team collectives
    KOKKOS_INLINE_FUNCTION void team_barrier() const {
        m_item.barrier(sycl::access::fence_space::local_space);
    }

    template <class ValueType>
    KOKKOS_INLINE_FUNCTION void team_broadcast(ValueType& val,
                                               const int& thread_id) const {
        team_barrier(); // Wait for shared data write until all threads arrive here
        if(m_team_rank == thread_id){
            *((ValueType*)m_team_reduce) = val;
        }
        team_barrier(); // Wait for shared data read until root thread writes
        val = *((ValueType*)m_team_reduce);
    }

    template <class Closure, class ValueType>
    KOKKOS_INLINE_FUNCTION void team_broadcast(Closure const& f, ValueType& val,
                                               const int& thread_id) const {
        f(val);
        team_barrier(); // Wait for shared data write until all threads arrive here
        if(m_team_rank == thread_id){
            *((ValueType*)m_team_reduce) = val;
        }
        team_barrier(); // Wait for shared data read until root thread writes
        val = *((ValueType*)m_team_reduce);
    }

    template <typename ReducerType>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<is_reducer<ReducerType>::value>::type
    team_reduce(ReducerType const& reducer) const noexcept {
        team_reduce(reducer, reducer.reference());
    }

    template <typename ReducerType>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<is_reducer<ReducerType>::value>::type
    team_reduce(ReducerType const& reducer,
                typename ReducerType::value_type& value) const noexcept {
        (void)reducer;
        (void)value;
        team_barrier();
        using value_type = typename ReducerType::value_type;
        value_type* base_data = (value_type*)m_team_reduce;
        base_data[m_team_rank] = value;
        team_barrier();
        if(m_team_rank == 0){
            for(int i = 1; i < m_team_size; i++){
                reducer.join(base_data[0], base_data[i]);
            }
        }
        team_barrier();
        value = base_data[0];
    }

    template <typename ArgType>
    KOKKOS_INLINE_FUNCTION ArgType team_scan(const ArgType& value,
                                             ArgType* const global_accum) const {
        team_barrier();
        using value_type = typename ReducerType::value_type;
        value_type* base_data = (value_type*)m_team_reduce;
        if(m_team_rank == 0) base_data[0] = 0;
        base_data[m_team_rank + 1] = value;
        if(m_team_rank == 0){
            for(int i = 1; i <= m_team_size; i++){
                base_data[i] += base_data[i-1];
            }
            if(global_accum) *global_accum = base_data[m_team_size];
        }
        team_barrier();
        return base_data[m_team_rank];
    }

    template <typename ArgType>
    KOKKOS_INLINE_FUNCTION ArgType team_scan(const ArgType& value) const {
        return this->template team_scan<ArgType>(value, nullptr);
    }

};

//----------------------------------------------------------------------------
template <class... Properties>
class TeamPolicyInternal<Kokkos::SYCL, Properties...>
    : public PolicyTraits<Properties...> {
public:
    //! Tag this class as a kokkos execution policy
    using execution_policy = TeamPolicyInternal;
    using traits = PolicyTraits<Properties...>;
    using member_type = Kokkos::Impl::SYCLTeamMember;

    const typename traits::execution_space& space() const {
        static typename traits::execution_space space_;
        return space_;
    }

    template <class ExecSpace, class... OtherProperties>
    friend class TeamPolicyInternal;

private:
    typename traits::execution_space m_space;
    int m_league_size;
    int m_team_size;
    int m_vector_length;
    int m_team_scratch_size[2];
    int m_thread_scratch_size[2];
    int m_chunk_size;
    bool m_tune_team;
    bool m_tune_vector;

public:
    //! Execution space of this execution policy
    using execution_space = Kokkos::SYCL;

    template <class... OtherProperties>
    TeamPolicyInternal(const TeamPolicyInternal<OtherProperties...>& p) {
        m_league_size            = p.m_league_size;
        m_team_size              = p.m_team_size;
        m_vector_length          = p.m_vector_length;
        m_team_scratch_size[0]   = p.m_team_scratch_size[0];
        m_team_scratch_size[1]   = p.m_team_scratch_size[1];
        m_thread_scratch_size[0] = p.m_thread_scratch_size[0];
        m_thread_scratch_size[1] = p.m_thread_scratch_size[1];
        m_chunk_size             = p.m_chunk_size;
        m_space                  = p.m_space;
        m_tune_team              = p.m_tune_team;
        m_tune_vector            = p.m_tune_vector;
    }

    TeamPolicyInternal()
            : m_space(typename traits::execution_space()),
              m_league_size(0),
              m_team_size(-1),
              m_vector_length(0),
              m_team_scratch_size{0, 0},
              m_thread_scratch_size{0, 0},
              m_chunk_size(0),
              m_tune_team(false),
              m_tune_vector(false) {}

    /** \brief  Specify league size, specify team size, specify vector length */
    TeamPolicyInternal(const execution_space space_, int league_size_,
                       int team_size_request, int vector_length_request = 1)
            : m_space(space_),
              m_league_size(league_size_),
              m_team_size(team_size_request),
              m_vector_length(vector_length_request),
              m_team_scratch_size{0, 0},
              m_thread_scratch_size{0, 0},
              m_chunk_size(0),
              m_tune_team(bool(team_size_request<=0)),
              m_tune_vector(bool(vector_length_request<=0)) {
//        using namespace cl::sycl::info;
//        if(league_size_ > m_space.impl_internal_space_instance()->m_queue->get_device().template get_info<device::max_work_group_size>()){
//            Impl::throw_runtime_exception(
//                    "Requested too large league_size for TeamPolicy on SYCL execution "
//                    "space.");
//        }
    }

    /** \brief  Specify league size, request team size, specify vector length */
    TeamPolicyInternal(const execution_space space_, int league_size_,
                       const Kokkos::AUTO_t& /* team_size_request */
                        ,
                       int vector_length_request = 1)
            : TeamPolicyInternal(space_, league_size_, -1, vector_length_request) {}

    /** \brief  Specify league size, request team size and vector length */
    TeamPolicyInternal(const execution_space space_, int league_size_,
                       const Kokkos::AUTO_t& /* team_size_request */,
                       const Kokkos::AUTO_t& /* vector_length_request */
                        )
            : TeamPolicyInternal(space_, league_size_, -1, -1) {}

    /** \brief  Specify league size, specify team size, request vector length */
    TeamPolicyInternal(const execution_space space_, int league_size_,
                       int team_size_request, const Kokkos::AUTO_t&)
            : TeamPolicyInternal(space_, league_size_, team_size_request, -1) {}

    TeamPolicyInternal(int league_size_, int team_size_request,
                       int vector_length_request = 1)
            : TeamPolicyInternal(typename traits::execution_space(), league_size_,
                                 team_size_request, vector_length_request) {}

    TeamPolicyInternal(int league_size_, const Kokkos::AUTO_t& team_size_request,
                       int vector_length_request = 1)
            : TeamPolicyInternal(typename traits::execution_space(), league_size_,
                                 team_size_request, vector_length_request) {}

    /** \brief  Specify league size, request team size */
    TeamPolicyInternal(int league_size_, const Kokkos::AUTO_t& team_size_request,
                       const Kokkos::AUTO_t& vector_length_request)
            : TeamPolicyInternal(typename traits::execution_space(), league_size_,
                                 team_size_request, vector_length_request) {}

    /** \brief  Specify league size, request team size */
    TeamPolicyInternal(int league_size_, int team_size_request,
                       const Kokkos::AUTO_t& vector_length_request)
            : TeamPolicyInternal(typename traits::execution_space(), league_size_,
                                 team_size_request, vector_length_request) {}

    /** \brief set chunk_size to a discrete value*/
    inline TeamPolicyInternal& set_chunk_size(
            typename traits::index_type chunk_size_) {
        m_chunk_size = chunk_size_;
        return *this;
    }

    /** \brief set per team scratch size for a specific level of the scratch
     * hierarchy */
    inline TeamPolicyInternal& set_scratch_size(const int& level,
                                                const PerTeamValue& per_team) {
        m_team_scratch_size[level] = per_team.value;
        return *this;
    }

    /** \brief set per thread scratch size for a specific level of the scratch
     * hierarchy */
    inline TeamPolicyInternal& set_scratch_size(
            const int& level, const PerThreadValue& per_thread) {
        m_thread_scratch_size[level] = per_thread.value;
        return *this;
    }

    /** \brief set per thread and per team scratch size for a specific level of
     * the scratch hierarchy */
    inline TeamPolicyInternal& set_scratch_size(
            const int& level, const PerTeamValue& per_team,
            const PerThreadValue& per_thread) {
        m_team_scratch_size[level]   = per_team.value;
        m_thread_scratch_size[level] = per_thread.value;
        return *this;
    }

    template <class FunctorType>
    int team_size_max(const FunctorType&, const ParallelForTag&) const {
        using namespace cl::sycl::info;
        return m_space.impl_internal_space_instance()->m_queue->get_device().template get_info<device::max_work_group_size>();
    }
    template <class FunctorType>
    int team_size_max(const FunctorType&, const ParallelReduceTag&) const {
        using namespace cl::sycl::info;
        return m_space.impl_internal_space_instance()->m_queue->get_device().template get_info<device::max_work_group_size>();
    }
    template <class FunctorType, class ReducerType>
    inline int team_size_max(const FunctorType& f, const ReducerType&,
                             const ParallelReduceTag& t) const {
        return team_size_max(f, t);
    }

    template <class FunctorType>
    int team_size_recommended(const FunctorType&, const ParallelForTag&) const {
        return 32;
    }
    template <class FunctorType>
    int team_size_recommended(const FunctorType&,
                              const ParallelReduceTag&) const {
        return 32;
    }
    template <class FunctorType, class ReducerType>
    inline int team_size_recommended(const FunctorType& f, const ReducerType&,
                                     const ParallelReduceTag& t) const {
        return team_size_recommended(f, t);
    }

    inline int team_size() const { return m_team_size; }
    inline int league_size() const { return m_league_size; }
    inline int scratch_size(int level, int team_size_ = -1) const {
        if (team_size_ < 0) team_size_ = m_team_size;
        return m_team_scratch_size[level] +
               team_size_ * m_thread_scratch_size[level];
    }
    inline int team_scratch_size(int level) const {
        return m_team_scratch_size[level];
    }
    inline int thread_scratch_size(int level) const {
        return m_thread_scratch_size[level];
    }
    inline int chunk_size() const {
        return m_chunk_size;
    }

    inline int impl_vector_length() const { return m_vector_length; }

};

}
}

#endif //MY_KOKKOS_KOKKOS_SYCL_TEAM_HPP
