//
// Created by Ryanxiejh on 2021/2/22.
//

#include <Kokkos_HostSpace.hpp>
#include "../Kokkos_SYCL.hpp"
#include <Kokkos_SYCL_Space.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include <impl/Kokkos_MemorySpace.hpp>
#include <impl/Kokkos_Profiling.hpp>

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
namespace Kokkos {
namespace Impl {
namespace {
auto USM_memcpy(cl::sycl::queue& q, void* dst, const void* src, size_t n) {
  return q.memcpy(dst, src, n);
}

void USM_memcpy(Kokkos::Impl::SYCLInternal& space, void* dst,
                const void* src, size_t n) {
  (void)USM_memcpy(*space.m_queue, dst, src, n);
}

void USM_memcpy(void* dst, const void* src, size_t n) {
  Kokkos::Impl::SYCLInternal::singleton().m_queue->wait();
  USM_memcpy(*Kokkos::Impl::SYCLInternal::singleton().m_queue,
             dst, src, n)
      .wait();
}
}  // namespace

DeepCopy<Kokkos::SYCLSpace,
         Kokkos::SYCLSpace, Kokkos::SYCL>::
    DeepCopy(const Kokkos::SYCL& instance, void* dst,
             const void* src, size_t n) {
  USM_memcpy(*instance.impl_internal_space_instance(), dst, src, n);
}

DeepCopy<Kokkos::SYCLSpace,
         Kokkos::SYCLSpace,
         Kokkos::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}

DeepCopy<Kokkos::HostSpace, Kokkos::SYCLSpace,
         Kokkos::SYCL>::DeepCopy(const Kokkos::SYCL&
                                                   instance,
                                               void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(*instance.impl_internal_space_instance(), dst, src, n);
}

DeepCopy<Kokkos::HostSpace, Kokkos::SYCLSpace,
         Kokkos::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}

DeepCopy<Kokkos::SYCLSpace, Kokkos::HostSpace,
         Kokkos::SYCL>::DeepCopy(const Kokkos::SYCL&
                                                   instance,
                                               void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(*instance.impl_internal_space_instance(), dst, src, n);
}

DeepCopy<Kokkos::SYCLSpace, Kokkos::HostSpace,
         Kokkos::SYCL>::DeepCopy(void* dst, const void* src,
                                               size_t n) {
  USM_memcpy(dst, src, n);
}

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

SYCLSpace::SYCLSpace() : m_device(SYCL().sycl_device()) {}

void* SYCLSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}
void* SYCLSpace::allocate(const char* arg_label,
                                   const size_t arg_alloc_size,
                                   const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}

void* SYCLSpace::impl_allocate(
    const char* arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  const cl::sycl::queue& queue =
      *SYCL().impl_internal_space_instance()->m_queue;
  void* const hostPtr = cl::sycl::malloc_device(arg_alloc_size, queue);

  if (hostPtr == nullptr)
    throw RawMemoryAllocationFailure(
        arg_alloc_size, 1, RawMemoryAllocationFailure::FailureMode::Unknown,
        RawMemoryAllocationFailure::AllocationMechanism::SYCLMalloc);

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, hostPtr,
                                    reported_size);
  }

  return hostPtr;
}

void SYCLSpace::deallocate(void* const arg_alloc_ptr,
                                    const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}
void SYCLSpace::deallocate(const char* arg_label,
                                    void* const arg_alloc_ptr,
                                    const size_t arg_alloc_size,
                                    const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void SYCLSpace::impl_deallocate(
    const char* arg_label, void* const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }
  const cl::sycl::queue& queue =
      *SYCL().impl_internal_space_instance()->m_queue;
  cl::sycl::free(arg_alloc_ptr, queue);
}

}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_ENABLE_DEBUG
SharedAllocationRecord<void, void> SharedAllocationRecord<
    Kokkos::SYCLSpace, void>::s_root_record;
#endif

SharedAllocationRecord<Kokkos::SYCLSpace, void>::
    SharedAllocationRecord(
        const Kokkos::SYCLSpace& space,
        const std::string& label, const size_t size,
        const SharedAllocationRecord<void, void>::function_type dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_ENABLE_DEBUG
          &SharedAllocationRecord<Kokkos::SYCLSpace,
                                  void>::s_root_record,
#endif
          Kokkos::Impl::checked_allocation_with_header(space, label, size),
          sizeof(SharedAllocationHeader) + size, dealloc),
      m_space(space) {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(
        Kokkos::Profiling::make_space_handle(space.name()), label, data(),
        size);
  }

  SharedAllocationHeader header;

  // Fill in the Header information
  header.m_record = static_cast<SharedAllocationRecord<void, void>*>(this);

  strncpy(header.m_label, label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  header.m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;

  // Copy to device memory
  Kokkos::Impl::DeepCopy<Kokkos::SYCLSpace, HostSpace>(
      RecordBase::m_alloc_ptr, &header, sizeof(SharedAllocationHeader));
}

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

std::string SharedAllocationRecord<Kokkos::SYCLSpace,
                                   void>::get_label() const {
  SharedAllocationHeader header;

  Kokkos::Impl::DeepCopy<Kokkos::HostSpace,
                         Kokkos::SYCLSpace>(
      &header, RecordBase::head(), sizeof(SharedAllocationHeader));

  return std::string(header.m_label);
}

SharedAllocationRecord<Kokkos::SYCLSpace, void>*
SharedAllocationRecord<Kokkos::SYCLSpace, void>::
    allocate(const Kokkos::SYCLSpace& arg_space,
             const std::string& arg_label, const size_t arg_alloc_size) {
  return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
}

void SharedAllocationRecord<Kokkos::SYCLSpace, void>::
    deallocate(SharedAllocationRecord<void, void>* arg_rec) {
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord<Kokkos::SYCLSpace,
                       void>::~SharedAllocationRecord() {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    SharedAllocationHeader header;
    Kokkos::Impl::DeepCopy<Kokkos::SYCLSpace,
                           Kokkos::HostSpace>(&header, RecordBase::m_alloc_ptr,
                                              sizeof(SharedAllocationHeader));

    Kokkos::Profiling::deallocateData(
        Kokkos::Profiling::make_space_handle(
            Kokkos::SYCLSpace::name()),
        header.m_label, data(), size());
  }

  m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

//----------------------------------------------------------------------------

void* SharedAllocationRecord<Kokkos::SYCLSpace, void>::
    allocate_tracked(const Kokkos::SYCLSpace& arg_space,
                     const std::string& arg_alloc_label,
                     const size_t arg_alloc_size) {
  if (!arg_alloc_size) return nullptr;

  SharedAllocationRecord* const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  RecordBase::increment(r);

  return r->data();
}

void SharedAllocationRecord<Kokkos::SYCLSpace,
                            void>::deallocate_tracked(void* const
                                                          arg_alloc_ptr) {
  if (arg_alloc_ptr != nullptr) {
    SharedAllocationRecord* const r = get_record(arg_alloc_ptr);

    RecordBase::decrement(r);
  }
}

void* SharedAllocationRecord<Kokkos::SYCLSpace, void>::
    reallocate_tracked(void* const arg_alloc_ptr, const size_t arg_alloc_size) {
  SharedAllocationRecord* const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord* const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<Kokkos::SYCLSpace,
                         Kokkos::SYCLSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);

  return r_new->data();
}

//----------------------------------------------------------------------------

SharedAllocationRecord<Kokkos::SYCLSpace, void>*
SharedAllocationRecord<Kokkos::SYCLSpace,
                       void>::get_record(void* alloc_ptr) {
  using Header = SharedAllocationHeader;
  using RecordSYCL =
      SharedAllocationRecord<Kokkos::SYCLSpace, void>;

  // Copy the header from the allocation
  Header head;

  Header const* const head_sycl =
      alloc_ptr ? Header::get_header(alloc_ptr) : nullptr;

  if (alloc_ptr) {
    Kokkos::Impl::DeepCopy<Kokkos::HostSpace,
                           Kokkos::SYCLSpace>(
        &head, head_sycl, sizeof(SharedAllocationHeader));
  }

  RecordSYCL* const record =
      alloc_ptr ? static_cast<RecordSYCL*>(head.m_record) : nullptr;

  if (!alloc_ptr || record->m_alloc_ptr != head_sycl) {
    Kokkos::Impl::throw_runtime_exception(
        std::string("Kokkos::Impl::SharedAllocationRecord< "
                    "Kokkos::SYCLSpace "
                    ", void >::get_record ERROR"));
  }

  return record;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord<Kokkos::SYCLSpace, void>::
    print_records(std::ostream& s,
                  const Kokkos::SYCLSpace&,
                  bool detail) {
#ifdef KOKKOS_ENABLE_DEBUG
  SharedAllocationRecord<void, void>* r = &s_root_record;

  char buffer[256];

  SharedAllocationHeader head;

  if (detail) {
    do {
      if (r->m_alloc_ptr) {
        Kokkos::Impl::DeepCopy<Kokkos::HostSpace,
                               Kokkos::SYCLSpace>(
            &head, r->m_alloc_ptr, sizeof(SharedAllocationHeader));
      } else {
        head.m_label[0] = 0;
      }

      // Formatting dependent on sizeof(uintptr_t)
      const char* format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) {
        format_string =
            "SYCL addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx "
            "+ %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
        format_string =
            "SYCL addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ "
            "0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf(buffer, 256, format_string, reinterpret_cast<uintptr_t>(r),
               reinterpret_cast<uintptr_t>(r->m_prev),
               reinterpret_cast<uintptr_t>(r->m_next),
               reinterpret_cast<uintptr_t>(r->m_alloc_ptr), r->m_alloc_size,
               r->m_count, reinterpret_cast<uintptr_t>(r->m_dealloc),
               head.m_label);
      s << buffer;
      r = r->m_next;
    } while (r != &s_root_record);
  } else {
    do {
      if (r->m_alloc_ptr) {
        Kokkos::Impl::DeepCopy<Kokkos::HostSpace,
                               Kokkos::SYCLSpace>(
            &head, r->m_alloc_ptr, sizeof(SharedAllocationHeader));

        // Formatting dependent on sizeof(uintptr_t)
        const char* format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string = "SYCL [ 0x%.12lx + %ld ] %s\n";
        } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string = "SYCL [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf(buffer, 256, format_string,
                 reinterpret_cast<uintptr_t>(r->data()), r->size(),
                 head.m_label);
      } else {
        snprintf(buffer, 256, "SYCL [ 0 + 0 ]\n");
      }
      s << buffer;
      r = r->m_next;
    } while (r != &s_root_record);
  }
#else
  (void)s;
  (void)detail;
  throw_runtime_exception(
      "Kokkos::Impl::SharedAllocationRecord<SYCLSpace>::print_records"
      " only works with KOKKOS_ENABLE_DEBUG enabled");
#endif
}

}  // namespace Impl
}  // namespace Kokkos
