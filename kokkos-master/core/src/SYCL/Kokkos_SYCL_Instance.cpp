//
// Created by Ryanxiejh on 2021/2/22.
//

#include <Kokkos_Concepts.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include "../Kokkos_SYCL.hpp"
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Serial.hpp>
#include <impl/Kokkos_Error.hpp>

namespace Kokkos {
namespace Impl {

int SYCLInternal::was_finalized = 0;

SYCLInternal::~SYCLInternal() {
  if (m_scratchSpace || m_scratchFlags) {
    std::cerr << "Kokkos::SYCL ERROR: Failed to call "
                 "Kokkos::SYCL::finalize()"
              << std::endl;
    std::cerr.flush();
  }

  m_scratchSpace = nullptr;
  m_scratchFlags = nullptr;
}

int SYCLInternal::verify_is_initialized(const char* const label) const {
  if (!is_initialized()) {
    std::cerr << "Kokkos::SYCL::" << label
              << " : ERROR device not initialized" << std::endl;
  }
  return is_initialized();
}
SYCLInternal& SYCLInternal::singleton() {
  static SYCLInternal self;
  return self;
}

// FIME_SYCL
void SYCLInternal::initialize(const sycl::device& d) {
  if (was_finalized)
    Kokkos::abort("Calling SYCL::initialize after SYCL::finalize is illegal\n");

  if (is_initialized()) return;

  if (!HostSpace::execution_space::impl_is_initialized()) {
    const std::string msg(
        "SYCL::initialize ERROR : HostSpace::execution_space is not "
        "initialized");
    Kokkos::Impl::throw_runtime_exception(msg);
  }

  const bool ok_init = nullptr == m_scratchSpace || nullptr == m_scratchFlags;
  const bool ok_dev  = true;
  if (ok_init && ok_dev) {
    m_queue = std::make_unique<sycl::queue>(d);
  } else {
    std::ostringstream msg;
    msg << "Kokkos::SYCL::initialize(...) FAILED";

    if (!ok_init) {
      msg << " : Already initialized";
    }
    Kokkos::Impl::throw_runtime_exception(msg.str());
  }
}

void SYCLInternal::finalize() {
  SYCL().fence();
  was_finalized = 1;
  if (nullptr != m_scratchSpace || nullptr != m_scratchFlags) {
    // FIXME_SYCL
    std::abort();
  }

  m_queue.reset();
}

}  // namespace Impl
}  // namespace Kokkos
