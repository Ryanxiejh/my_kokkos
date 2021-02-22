//
// Created by Ryanxiejh on 2021/2/22.
//

#ifndef KOKKOS_SYCL_HPP
#define KOKKOS_SYCL_HPP

#include "Kokkos_Macros.hpp"

#ifdef KOKKOS_ENABLE_SYCL
#include <CL/sycl.hpp>
#include <Kokkos_SYCL_Space.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <impl/Kokkos_ExecSpaceInitializer.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

namespace Kokkos {
namespace Impl {
class SYCLInternal;
}

/// \class SYCL
/// \brief Kokkos device for multicore processors in the host memory space.
class SYCL {
 public:
  //------------------------------------
  //! \name Type declarations that all Kokkos devices must provide.
  //@{

  //! Tag this class as a kokkos execution space
  using execution_space = SYCL;
  using memory_space    = SYCLSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  using array_layout = LayoutLeft;
  using size_type    = memory_space::size_type;

  using scratch_memory_space = ScratchMemorySpace<SYCL>;

  ~SYCL() = default;
  SYCL();

  SYCL(SYCL&&)      = default;
  SYCL(const SYCL&) = default;
  SYCL& operator=(SYCL&&) = default;
  SYCL& operator=(const SYCL&) = default;

  uint32_t impl_instance_id() const noexcept { return 0; }

  //@}
  //------------------------------------
  //! \name Functions that all Kokkos devices must implement.
  //@{

  KOKKOS_INLINE_FUNCTION static int in_parallel() {
#if defined(__SYCL_ARCH__)
    return true;
#else
    return false;
#endif
  }

  /** \brief  Set the device in a "sleep" state. */
  static bool sleep();

  /** \brief Wake the device from the 'sleep' state. A noop for OpenMP. */
  static bool wake();

  /** \brief Wait until all dispatched functors complete. A noop for OpenMP. */
  static void impl_static_fence();
  void fence() const;

  /// \brief Print configuration information to the given output stream.
  static void print_configuration(std::ostream&, const bool detail = false);

  /// \brief Free any resources being consumed by the device.
  static void impl_finalize();

  sycl::device m_device;
  static void impl_initialize(const sycl::device_selector& selector);

  int sycl_device() const;

  static bool impl_is_initialized();

  static int concurrency();
  static const char* name();

  inline Impl::SYCLInternal* impl_internal_space_instance() const {
    return m_space_instance;
  }

 private:
  Impl::SYCLInternal* m_space_instance;
};

namespace Impl {

class SYCLSpaceInitializer : public Kokkos::Impl::ExecSpaceInitializerBase {
 public:
  void initialize(const InitArguments& args) final;
  void finalize(const bool) final;
  void fence() final;
  void print_configuration(std::ostream& msg, const bool detail) final;
};

}  // namespace Impl

namespace Tools {
namespace Experimental {
template <>
struct DeviceTypeTraits<Kokkos::SYCL> {
  /// \brief An ID to differentiate (for example) Serial from OpenMP in Tooling
  static constexpr DeviceType id = DeviceType::SYCL;
};
}  // namespace Experimental
}  // namespace Tools

}  // namespace Kokkos

#endif
#endif
