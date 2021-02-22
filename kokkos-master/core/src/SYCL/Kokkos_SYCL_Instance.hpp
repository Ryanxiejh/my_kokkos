//
// Created by Ryanxiejh on 2021/2/22.
//

#ifndef KOKKOS_SYCL_INSTANCE_HPP_
#define KOKKOS_SYCL_INSTANCE_HPP_

#include <memory>
#include <CL/sycl.hpp>

namespace Kokkos {
namespace Impl {

class SYCLInternal {
 public:
  using size_type = int;

  SYCLInternal() = default;
  ~SYCLInternal();

  SYCLInternal(const SYCLInternal&) = delete;
  SYCLInternal& operator=(const SYCLInternal&) = delete;
  SYCLInternal& operator=(SYCLInternal&&) = delete;
  SYCLInternal(SYCLInternal&&)            = delete;

  int m_syclDev             = -1;
  size_type* m_scratchSpace = nullptr;
  size_type* m_scratchFlags = nullptr;

  std::unique_ptr<cl::sycl::queue> m_queue;

  static int was_finalized;

  static SYCLInternal& singleton();

  int verify_is_initialized(const char* const label) const;

  void initialize(const cl::sycl::device& d);

  int is_initialized() const { return m_queue != nullptr; }

  void finalize();
};

}  // namespace Impl
}  // namespace Kokkos
#endif
