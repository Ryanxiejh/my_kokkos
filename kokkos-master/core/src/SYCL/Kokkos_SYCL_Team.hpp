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
        reducer.reference() = base_data[0];
        value = base_data[0];
    }

    template <typename ArgType>
    KOKKOS_INLINE_FUNCTION ArgType team_scan(const ArgType& value,
                                             ArgType* const global_accum) const {
        team_barrier();
        ArgType* base_data = (ArgType*)m_team_reduce;
        if(m_team_rank == 0) base_data[0] = ArgType{};
        base_data[m_team_rank + 1] = value;
        team_barrier();
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

} //namespace Impl
} //namespace Kokkos


namespace Kokkos{

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamThreadRangeBoundariesStruct<iType, Impl::SYCLTeamMember>
    TeamThreadRange(const Impl::SYCLTeamMember& thread,
                    const iType& count) {
  return Impl::TeamThreadRangeBoundariesStruct<iType,
                                               Impl::SYCLTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamThreadRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type,
    Impl::SYCLTeamMember>
TeamThreadRange(const Impl::SYCLTeamMember& thread, const iType1& begin,
                const iType2& end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamThreadRangeBoundariesStruct<iType,
                                               Impl::SYCLTeamMember>(
      thread, iType(begin), iType(end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>
    ThreadVectorRange(const Impl::SYCLTeamMember& thread,
                      const iType& count) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType,
                                                 Impl::SYCLTeamMember>(
      thread, count);
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>
    ThreadVectorRange(const Impl::SYCLTeamMember& thread,
                      const iType& arg_begin, const iType& arg_end) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType,
                                                 Impl::SYCLTeamMember>(
      thread, arg_begin, arg_end);
}

KOKKOS_INLINE_FUNCTION
Impl::ThreadSingleStruct<Impl::SYCLTeamMember> PerTeam(
    const Impl::SYCLTeamMember& thread) {
  return Impl::ThreadSingleStruct<Impl::SYCLTeamMember>(thread);
}

KOKKOS_INLINE_FUNCTION
Impl::VectorSingleStruct<Impl::SYCLTeamMember> PerThread(
    const Impl::SYCLTeamMember& thread) {
  return Impl::VectorSingleStruct<Impl::SYCLTeamMember>(thread);
}

} //namespace Kokkos

namespace Kokkos{

/** \brief  Inter-thread parallel_for.
 *
 *  Executes closure(iType i) for each i=[0..N).
 *
 * The range [0..N) is mapped to all threads of the the calling thread team.
 */
template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::SYCLTeamMember>&
        loop_boundaries,
    const Closure& closure) {
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment)
    closure(i);
}

/** \brief  Inter-thread parallel_reduce assuming summation.
 *
 *  Executes closure(iType i, ValueType & val) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all threads of the
 *  calling thread team and a summation of val is
 *  performed and put into result.
 */
template <typename iType, class Closure, typename ValueType>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<!Kokkos::is_reducer<ValueType>::value>::type
    parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                        iType, Impl::SYCLTeamMember>& loop_boundaries,
                    const Closure& closure, ValueType& result) {
  ValueType intermediate;
  Sum<ValueType> sum(intermediate);
  sum.init(intermediate);

  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    ValueType tmp = ValueType();
    closure(i, tmp);
    intermediate += tmp;
  }

  loop_boundaries.thread.team_reduce(sum, intermediate);
  result = sum.reference();
}

/** \brief  Inter-thread parallel_reduce with a reducer.
 *
 *  Executes closure(iType i, ValueType & val) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all threads of the
 *  calling thread team and a summation of val is
 *  performed and put into result.
 */
template <typename iType, class Closure, class ReducerType>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<Kokkos::is_reducer<ReducerType>::value>::type
    parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                        iType, Impl::SYCLTeamMember>& loop_boundaries,
                    const Closure& closure, const ReducerType& reducer) {
  typename ReducerType::value_type value;
  reducer.init(value);

  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    closure(i, value);
  }

  loop_boundaries.member.team_reduce(reducer, value);
}

/** \brief  Inter-thread parallel exclusive prefix sum.
 *
 *  Executes closure(iType i, ValueType & val, bool final) for each i=0..N-1.
 *
 *  The range [0..N) is mapped to each rank in the team (whose global rank is
 *  less than N) and a scan operation is performed.
 */
template <typename iType, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::SYCLTeamMember>& loop_bounds,
    const FunctorType& closure) {
  using value_type = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN, void,
      FunctorType>::value_type;

  auto scan_val = value_type{};

  // Intra-member scan
  for (iType i = loop_bounds.start; i < loop_bounds.end;
       i += loop_bounds.increment) {
      closure(i, scan_val, false);
  }

  // 'scan_val' output is the exclusive prefix sum
  scan_val = loop_bounds.thread.team_scan(scan_val);

  for (iType i = loop_bounds.start; i < loop_bounds.end;
       i += loop_bounds.increment) {
      closure(i, scan_val, true);
  }
}

} //namespace Kokkos

namespace Kokkos{

/** \brief  Inter-thread parallel_for.
 *
 *  Executes closure(iType i) for each i=[0..N).
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread.
 */
template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::SYCLTeamMember>& loop_boundaries,
    const Closure& closure) {
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment)
      closure(i);
}

/** \brief  Inter-thread parallel_reduce assuming summation.
 *
 *  Executes closure(iType i, ValueType & val) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all vector lanes of the
 *  calling thread and a summation of val is
 *  performed and put into result.
 */
template <typename iType, class Closure, typename ValueType>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<!Kokkos::is_reducer<ValueType>::value>::type
    parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                        iType, Impl::SYCLTeamMember>& loop_boundaries,
                    const Closure& closure, ValueType& result) {
  result = ValueType();
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    closure(i, result);
  }
}

/** \brief  Inter-thread parallel_reduce with a reducer.
 *
 *  Executes closure(iType i, ValueType & val) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all vector lanes of the
 *  calling thread and a summation of val is
 *  performed and put into result.
 */
template <typename iType, class Closure, typename ReducerType>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<Kokkos::is_reducer<ReducerType>::value>::type
    parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                        iType, Impl::SYCLTeamMember>& loop_boundaries,
                    const Closure& closure, const ReducerType& reducer) {
  reducer.init(reducer.reference());
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    closure(i, reducer.reference());
  }
}

/** \brief  Intra-thread vector parallel exclusive prefix sum.
 *
 *  Executes closure(iType i, ValueType & val, bool final) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all vector lanes in the
 *  thread and a scan operation is performed.
 *
 *  The range i=0..N-1 is mapped to all vector lanes in the thread and a scan
 *  operation is performed. Depending on the target execution space the operator
 *  might be called twice: once with final=false and once with final=true. When
 *  final==true val contains the prefix sum value. The contribution of this "i"
 *  needs to be added to val no matter whether final==true or not. In a serial
 *  execution (i.e. team_size==1) the operator is only called once with
 *  final==true. Scan_val will be set to the final sum value over all vector
 *  lanes.
 */
template <typename iType, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::SYCLTeamMember>& loop_boundaries,
    const FunctorType& closure) {
  using ValueTraits = Kokkos::Impl::FunctorValueTraits<FunctorType, void>;
  using value_type  = typename ValueTraits::value_type;

  value_type scan_val = value_type();

  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
      closure(i, scan_val, true);
  }
}

} //namespace Kokkos

namespace Kokkos{

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<
        Impl::SYCLTeamMember>& /*single_struct*/,
    const FunctorType& closure) {
    closure();
}

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<Impl::SYCLTeamMember>& single_struct,
    const FunctorType& closure) {
  if (single_struct.team_member.team_rank() == 0) closure();
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<
        Impl::SYCLTeamMember>& /*single_struct*/,
    const FunctorType& closure, ValueType& val) {
    closure(val);
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<Impl::SYCLTeamMember>& single_struct,
    const FunctorType& closure, ValueType& val) {
  if (single_struct.team_member.team_rank() == 0) {
      closure(val);
  }
  single_struct.team_member.team_broadcast(val, 0);
}

} //namespace Kokkos

#endif //MY_KOKKOS_KOKKOS_SYCL_TEAM_HPP
