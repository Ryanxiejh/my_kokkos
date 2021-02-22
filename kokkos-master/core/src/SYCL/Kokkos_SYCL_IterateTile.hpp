//
// Created by Ryanxiejh on 2021/2/22.
//

#ifndef MY_KOKKOS_KOKKOS_SYCL_ITERATETILE_HPP
#define MY_KOKKOS_KOKKOS_SYCL_ITERATETILE_HPP

namespace Kokkos{
namespace Impl{

template <int N, typename RP, typename Functor, typename Tag>
struct apply_impl;

// Rank 2
// Specializations for void tag type
template <typename RP, typename Functor>
struct apply_impl<2, RP, Functor, void> {
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
      : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

  inline void exec_range() const {
    // LL
    if (RP::inner_direction == RP::Left) {
        for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                m_func(dim0+m_offset[0],dim1+m_offset[1]);
            }
        }
    }
    // LR
    else {
        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
            for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                m_func(dim0+m_offset[0],dim1+m_offset[1]);
            }
        }
    }

  }  // end exec_range

 private:
  //const RP& m_rp;
  const Functor& m_func;
  const point_type& m_offset;
  const point_type& m_extent;
};

// Specializations for tag type
template <typename RP, typename Functor, typename Tag>
struct apply_impl<2, RP, Functor, Tag> {
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
      : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

  inline void exec_range() const {
    // LL
    if (RP::inner_direction == RP::Left) {
        for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                m_func(Tag(),dim0+m_offset[0],dim1+m_offset[1]);
            }
        }
    }
    // LR
    else {
        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
            for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                m_func(Tag(),dim0+m_offset[0],dim1+m_offset[1]);
            }
        }
    }

  }  // end exec_range

 private:
  //const RP& m_rp;
  const Functor& m_func;
  const point_type& m_offset;
  const point_type& m_extent;
};

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
template <typename RP, typename Functor, typename Tag,
          typename ValueType = void, typename Enable = void>
struct SyclIterateTile;

template <typename RP, typename Functor, typename Tag, typename ValueType>
struct SyclIterateTile<
    RP, Functor, Tag, ValueType, typename std::enable_if<is_void_type<ValueType>::value>::type> {
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  using value_type = void;

  SyclIterateTile() = default;
  /*inline*/ SyclIterateTile(RP const& rp, Functor const& func)
      : m_rp(rp), m_func(func) {}

  inline bool check_iteration_bounds(point_type& partial_tile,
                                     point_type& offset) const {
    bool is_full_tile = true;

    for (int i = 0; i < RP::rank; ++i) {
      if ((offset[i] + m_rp.m_tile[i]) <= m_rp.m_upper[i]) {
        partial_tile[i] = m_rp.m_tile[i];
      } else {
        is_full_tile = false;
        partial_tile[i] =
            (m_rp.m_upper[i] - 1 - offset[i]) == 0
                ? 1
                : (m_rp.m_upper[i] - m_rp.m_tile[i]) > 0
                      ? (m_rp.m_upper[i] - offset[i])
                      : (m_rp.m_upper[i] -
                         m_rp.m_lower[i]);  // when single tile encloses range
      }
    }

    return is_full_tile;
  }  // end check bounds

//  template <int Rank>
//  struct RankTag {
//    typedef RankTag type;
//    enum { value = (int)Rank };
//  };

  template <typename IType>
  inline void operator()(IType tile_idx) const {
    point_type m_offset;
    point_type m_tiledims;

    if (RP::outer_direction == RP::Left) {
      for (int i = 0; i < RP::rank; ++i) {
        m_offset[i] =
            (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i];
        tile_idx /= m_rp.m_tile_end[i];
      }
    } else {
      for (int i = RP::rank - 1; i >= 0; --i) {
        m_offset[i] =
            (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] + m_rp.m_lower[i];
        tile_idx /= m_rp.m_tile_end[i];
      }
    }

    // Check if offset+tiledim in bounds - if not, replace tile dims with the
    // partial tile dims
    const bool full_tile = check_iteration_bounds(m_tiledims, m_offset);

    apply_impl<RP::rank, RP, Functor, Tag>(m_func, m_offset, m_tiledims).exec_range();
  }

//  template <typename... Args>
//  typename std::enable_if<(sizeof...(Args) == RP::rank &&
//                           std::is_same<Tag, void>::value),
//                          void>::type
//  apply(Args&&... args) const {
//    m_func(args...);
//  }
//
//  template <typename... Args>
//  typename std::enable_if<(sizeof...(Args) == RP::rank &&
//                           !std::is_same<Tag, void>::value),
//                          void>::type
//  apply(Args&&... args) const {
//    m_func(m_tag, args...);
//  }

  const RP& m_rp;
  const Functor& m_func;
//  typename std::conditional<std::is_same<Tag, void>::value, int, Tag>::type
//      m_tag;
};


} //namespace Impl
} //namespace Kokkos

#endif //MY_KOKKOS_KOKKOS_SYCL_ITERATETILE_HPP
