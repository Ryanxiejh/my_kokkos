//
// Created by Ryanxiejh on 2021/2/22.
//

#ifndef MY_KOKKOS_KOKKOS_SYCL_ITERATETILE_HPP
#define MY_KOKKOS_KOKKOS_SYCL_ITERATETILE_HPP

namespace Kokkos{
namespace Impl{

template <int N, typename RP, typename Functor,typename ValueType, typename Tag>
struct apply_impl;

// Rank 1
// Specializations for void tag type parallel_for
template <typename RP, typename Functor>
struct apply_impl<1, RP, Functor, void, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {

        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
            m_func(dim0+m_offset[0]);
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
};

// Specializations for tag type parallel_for
template <typename RP, typename Functor, typename Tag>
struct apply_impl<1, RP, Functor, void, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {

        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
            m_func(Tag(), dim0+m_offset[0]);
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
};

// Specializations for void tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType>
struct apply_impl<1, RP, Functor, ValueType, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {

        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
            m_func(dim0+m_offset[0], m_v);
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Specializations for tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType, typename Tag>
struct apply_impl<1, RP, Functor, ValueType, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {

        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
            m_func(Tag(), dim0+m_offset[0], m_v);
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Rank 2
// Specializations for void tag type parallel_for
template <typename RP, typename Functor>
struct apply_impl<2, RP, Functor, void, void> {
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

// Specializations for tag type parallel_for
template <typename RP, typename Functor, typename Tag>
struct apply_impl<2, RP, Functor, void, Tag> {
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

// Specializations for void tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType>
struct apply_impl<2, RP, Functor, ValueType, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                    m_func(dim0+m_offset[0],dim1+m_offset[1],m_v);
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    m_func(dim0+m_offset[0],dim1+m_offset[1],m_v);
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Specializations for tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType, typename Tag>
struct apply_impl<2, RP, Functor, ValueType, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                    m_func(Tag(),dim0+m_offset[0],dim1+m_offset[1],m_v);
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    m_func(Tag(),dim0+m_offset[0],dim1+m_offset[1],m_v);
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Rank 3
// Specializations for void tag type parallel_for
template <typename RP, typename Functor>
struct apply_impl<3, RP, Functor, void, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                        m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2]);
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2]);
                    }
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

// Specializations for tag type parallel_for
template <typename RP, typename Functor, typename Tag>
struct apply_impl<3, RP, Functor, void, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                        m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2]);
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2]);
                    }
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

// Specializations for void tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType>
struct apply_impl<3, RP, Functor, ValueType, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                        m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], m_v);
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], m_v);
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Specializations for tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType, typename Tag>
struct apply_impl<3, RP, Functor, ValueType, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                        m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], m_v);
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], m_v);
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Rank 4
// Specializations for void tag type parallel_for
template <typename RP, typename Functor>
struct apply_impl<4, RP, Functor, void, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                    for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                            m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3]);
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3]);
                        }
                    }
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

// Specializations for tag type parallel_for
template <typename RP, typename Functor, typename Tag>
struct apply_impl<4, RP, Functor, void, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                    for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                            m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3]);
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3]);
                        }
                    }
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

// Specializations for void tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType>
struct apply_impl<4, RP, Functor, ValueType, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                    for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                            m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3], m_v);
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3], m_v);
                        }
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Specializations for tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType, typename Tag>
struct apply_impl<4, RP, Functor, ValueType, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                    for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                            m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3], m_v);
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3], m_v);
                        }
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Rank 5
// Specializations for void tag type parallel_for
template <typename RP, typename Functor>
struct apply_impl<5, RP, Functor, void, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                       dim4+m_offset[4]);
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                       dim4+m_offset[4]);
                            }
                        }
                    }
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

// Specializations for tag type parallel_for
template <typename RP, typename Functor, typename Tag>
struct apply_impl<5, RP, Functor, void, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                       dim4+m_offset[4]);
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                       dim4+m_offset[4]);
                            }
                        }
                    }
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

// Specializations for void tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType>
struct apply_impl<5, RP, Functor, ValueType, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                       dim4+m_offset[4], m_v);
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                       dim4+m_offset[4], m_v);
                            }
                        }
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Specializations for tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType, typename Tag>
struct apply_impl<5, RP, Functor, ValueType, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                       dim4+m_offset[4], m_v);
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                       dim4+m_offset[4], m_v);
                            }
                        }
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Rank 6
// Specializations for void tag type parallel_for
template <typename RP, typename Functor>
struct apply_impl<6, RP, Functor, void, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                    for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                        for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                            for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                    m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                           dim4+m_offset[4], dim5+m_offset[5]);
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                           dim4+m_offset[4], dim5+m_offset[5]);
                                }
                            }
                        }
                    }
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

// Specializations for tag type parallel_for
template <typename RP, typename Functor, typename Tag>
struct apply_impl<6, RP, Functor, void, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                    for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                        for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                            for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                    m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                           dim4+m_offset[4], dim5+m_offset[5]);
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                           dim4+m_offset[4], dim5+m_offset[5]);
                                }
                            }
                        }
                    }
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

// Specializations for void tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType>
struct apply_impl<6, RP, Functor, ValueType, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                    for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                        for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                            for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                    m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                           dim4+m_offset[4], dim5+m_offset[5], m_v);
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                           dim4+m_offset[4], dim5+m_offset[5], m_v);
                                }
                            }
                        }
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Specializations for tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType, typename Tag>
struct apply_impl<6, RP, Functor, ValueType, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                    for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                        for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                            for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                    m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                           dim4+m_offset[4], dim5+m_offset[5], m_v);
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                           dim4+m_offset[4], dim5+m_offset[5], m_v);
                                }
                            }
                        }
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Rank 7
// Specializations for void tag type parallel_for
template <typename RP, typename Functor>
struct apply_impl<7, RP, Functor, void, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                    for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                    for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                        m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                               dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                                        m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                               dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6]);
                                    }
                                }
                            }
                        }
                    }
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

// Specializations for tag type parallel_for
template <typename RP, typename Functor, typename Tag>
struct apply_impl<7, RP, Functor, void, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                    for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                    for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                        m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                               dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                                        m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                               dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6]);
                                    }
                                }
                            }
                        }
                    }
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

// Specializations for void tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType>
struct apply_impl<7, RP, Functor, ValueType, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                    for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                    for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                        m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                               dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], m_v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                                        m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                               dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], m_v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Specializations for tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType, typename Tag>
struct apply_impl<7, RP, Functor, ValueType, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                    for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                    for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                        m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                               dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], m_v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                                        m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                               dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], m_v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Rank 8
// Specializations for void tag type parallel_for
template <typename RP, typename Functor>
struct apply_impl<8, RP, Functor, void, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim7 = 0; dim7 < m_extent[7]; dim7++){
                for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                    for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                        for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                            for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                                for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                                    for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                            m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                                   dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], dim7+m_offset[7]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                                        for(index_type dim7 = 0; dim7 < m_extent[7]; dim7++){
                                            m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                                   dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], dim7+m_offset[7]);
                                        }
                                    }
                                }
                            }
                        }
                    }
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

// Specializations for tag type parallel_for
template <typename RP, typename Functor, typename Tag>
struct apply_impl<8, RP, Functor, void, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim7 = 0; dim7 < m_extent[7]; dim7++){
                for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                    for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                        for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                            for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                                for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                                    for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                            m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                                   dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], dim7+m_offset[7]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                                        for(index_type dim7 = 0; dim7 < m_extent[7]; dim7++){
                                            m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                                   dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], dim7+m_offset[7]);
                                        }
                                    }
                                }
                            }
                        }
                    }
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

// Specializations for void tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType>
struct apply_impl<8, RP, Functor, ValueType, void> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim7 = 0; dim7 < m_extent[7]; dim7++){
                for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                    for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                        for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                            for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                                for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                                    for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                            m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                                   dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], dim7+m_offset[7], m_v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                                        for(index_type dim7 = 0; dim7 < m_extent[7]; dim7++){
                                            m_func(dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                                   dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], dim7+m_offset[7], m_v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

// Specializations for tag type parallel_reduce(single value)
template <typename RP, typename Functor, typename ValueType, typename Tag>
struct apply_impl<8, RP, Functor, ValueType, Tag> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;
    using value_type = ValueType;

    apply_impl(/*const RP& rp_, */const Functor& f_, const point_type& offset_, const point_type& extent_, value_type& v)
            : /*m_rp(rp_), */m_func(f_), m_offset(offset_), m_extent(extent_), m_v(v) {}

    inline void exec_range() const {
        // LL
        if (RP::inner_direction == RP::Left) {
            for(index_type dim7 = 0; dim7 < m_extent[7]; dim7++){
                for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                    for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                        for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                            for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                                for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                                    for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                                        for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                                            m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                                   dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], dim7+m_offset[7], m_v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
            // LR
        else {
            for(index_type dim0 = 0; dim0 < m_extent[0]; dim0++){
                for(index_type dim1 = 0; dim1 < m_extent[1]; dim1++){
                    for(index_type dim2 = 0; dim2 < m_extent[2]; dim2++){
                        for(index_type dim3 = 0; dim3 < m_extent[3]; dim3++){
                            for(index_type dim4 = 0; dim4 < m_extent[4]; dim4++){
                                for(index_type dim5 = 0; dim5 < m_extent[5]; dim5++){
                                    for(index_type dim6 = 0; dim6 < m_extent[6]; dim6++){
                                        for(index_type dim7 = 0; dim7 < m_extent[7]; dim7++){
                                            m_func(Tag(), dim0+m_offset[0], dim1+m_offset[1], dim2+m_offset[2], dim3+m_offset[3],
                                                   dim4+m_offset[4], dim5+m_offset[5], dim6+m_offset[6], dim7+m_offset[7], m_v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }  // end exec_range

private:
    //const RP& m_rp;
    const Functor& m_func;
    const point_type& m_offset;
    const point_type& m_extent;
    ValueType& m_v;
};

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
template <typename RP, typename Functor, typename Tag,
        typename ValueType = void, typename Enable = void>
struct SyclIterateTile;

//parallel_for
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

        apply_impl<RP::rank, RP, Functor, value_type, Tag>(m_func, m_offset, m_tiledims).exec_range();
    }

    const RP& m_rp;
    const Functor& m_func;
};

//parallel_reduce: single value
template <typename RP, typename Functor, typename Tag, typename ValueType>
struct SyclIterateTile<
        RP, Functor, Tag, ValueType, typename std::enable_if<!is_void_type<ValueType>::value &&
                                                             !is_type_array<ValueType>::value>::type> {
    using index_type = typename RP::index_type;
    using point_type = typename RP::point_type;

    using value_type = ValueType;

    SyclIterateTile() = default;
    /*inline*/ SyclIterateTile(RP const& rp, Functor const& func, value_type& v)
            : m_rp(rp), m_func(func), m_v(v){}

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

        apply_impl<RP::rank, RP, Functor, value_type, Tag>(m_func, m_offset, m_tiledims, m_v).exec_range();
    }

    const RP& m_rp;
    const Functor& m_func;
    value_type& m_v;
};


} //namespace Impl
} //namespace Kokkos

#endif //MY_KOKKOS_KOKKOS_SYCL_ITERATETILE_HPP
