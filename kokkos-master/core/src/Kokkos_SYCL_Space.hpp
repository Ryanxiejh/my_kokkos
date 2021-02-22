//
// Created by Ryanxiejh on 2021/2/22.
//

#ifndef KOKKOS_SYCLSPACE_HPP
#define KOKKOS_SYCLSPACE_HPP

#include <Kokkos_Core_fwd.hpp>

#ifdef KOKKOS_ENABLE_SYCL
#include <Kokkos_Concepts.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_Tools.hpp>

namespace Kokkos {

class SYCLSpace {
 public:
  using execution_space = SYCL;
  using memory_space    = SYCLSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using size_type       = Impl::SYCLInternal::size_type;

  SYCLSpace();

  void* allocate(const std::size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  void deallocate(void* const arg_alloc_ptr,
                  const std::size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  template <class, class, class, class>
  friend class LogicalMemorySpace;
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

 public:
  static constexpr const char* name() { return "SYCLDeviceUSM"; };

 private:
  int m_device;
};

namespace Impl {
static_assert(Kokkos::Impl::MemorySpaceAccess<
                  Kokkos::SYCLSpace,
                  Kokkos::SYCLSpace>::assignable,
              "");

template <>
struct MemorySpaceAccess<Kokkos::HostSpace,
                         Kokkos::SYCLSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::SYCLSpace,
                         Kokkos::HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = false };
  enum : bool { deepcopy = true };
};

}  // namespace Impl

namespace Impl{

template <>
struct DeepCopy<Kokkos::SYCLSpace, Kokkos::SYCLSpace, Kokkos::SYCL> {
  DeepCopy(void* dst, const void* src, size_t);
  DeepCopy(const Kokkos::SYCL&, void* dst, const void* src, size_t);
};

template <>
struct DeepCopy<Kokkos::HostSpace, Kokkos::SYCLSpace, Kokkos::SYCL> {
  DeepCopy(void* dst, const void* src, size_t);
  DeepCopy(const Kokkos::SYCL&, void* dst, const void* src, size_t);
};

template <>
struct DeepCopy<Kokkos::SYCLSpace, Kokkos::HostSpace, Kokkos::SYCL> {
  DeepCopy(void* dst, const void* src, size_t);
  DeepCopy(const Kokkos::SYCL&, void* dst, const void* src, size_t);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::SYCLSpace, Kokkos::SYCLSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n);
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Kokkos::SYCLSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n);
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::SYCLSpace, Kokkos::HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n);
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n);
};

} //namespace Impl


namespace Impl {

template <>
class SharedAllocationRecord<Kokkos::SYCLSpace, void>
    : public SharedAllocationRecord<void, void> {
 private:
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static void deallocate(RecordBase*);

#ifdef KOKKOS_ENABLE_DEBUG
  static RecordBase s_root_record;
#endif

  const Kokkos::SYCLSpace m_space;

 protected:
  ~SharedAllocationRecord();

  SharedAllocationRecord(
      const Kokkos::SYCLSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

 public:
  std::string get_label() const;

  static SharedAllocationRecord* allocate(
      const Kokkos::SYCLSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size);

  /**\brief  Allocate tracked memory in the space */
  static void* allocate_tracked(
      const Kokkos::SYCLSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size);

  /**\brief  Reallocate tracked memory in the space */
  static void* reallocate_tracked(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size);

  /**\brief  Deallocate tracked memory in the space */
  static void deallocate_tracked(void* const arg_alloc_ptr);

  static SharedAllocationRecord* get_record(void* arg_alloc_ptr);

  static void print_records(std::ostream&,
                            const Kokkos::SYCLSpace&,
                            bool detail = false);
};

}  // namespace Impl

}  // namespace Kokkos

#endif
#endif
