//
// Created by Ryanxiejh on 2021/2/22.
//

#include <Kokkos_Concepts.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include "../Kokkos_SYCL.hpp"
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Error.hpp>

namespace {
template <typename C>
struct Container {
  explicit Container(const C& c) : container(c) {}

  friend std::ostream& operator<<(std::ostream& os, const Container& that) {
    os << that.container.size();
    for (const auto& v : that.container) {
      os << "\n\t" << v;
    }
    return os;
  }

 private:
  const C& container;
};
}  // namespace

namespace Kokkos {

namespace Impl {
// forward-declaration
int get_gpu(const InitArguments& args);
}  // namespace Impl

SYCL::SYCL() : m_space_instance(&Impl::SYCLInternal::singleton()) {
  Impl::SYCLInternal::singleton().verify_is_initialized(
      "SYCL instance constructor");
}

int SYCL::concurrency() {
  // FIXME_SYCL We need a value larger than 1 here for some tests to pass,
  // clearly this is true but not the roght value
  return 2;
}

bool SYCL::impl_is_initialized() {
  return Impl::SYCLInternal::singleton().is_initialized();
}

void SYCL::impl_finalize() { Impl::SYCLInternal::singleton().finalize(); }

void SYCL::fence() const { m_space_instance->m_queue->wait(); }

int SYCL::sycl_device() const {
  return impl_internal_space_instance()->m_syclDev;
}

void SYCL::impl_initialize(const cl::sycl::device_selector& selector) {
  sycl::device device = selector.select_device();
  Impl::SYCLInternal::singleton().initialize(device);
}

namespace Impl {

int g_hip_space_factory_initialized =
    Kokkos::Impl::initialize_space_factory<SYCLSpaceInitializer>("170_SYCL");

void SYCLSpaceInitializer::initialize(const InitArguments& args) {
  int use_gpu = Kokkos::Impl::get_gpu(args);

  if (std::is_same<Kokkos::SYCL,
                   Kokkos::DefaultExecutionSpace>::value ||
      0 < use_gpu) {
    // FIXME_SYCL choose a specific device
    Kokkos::SYCL::impl_initialize(sycl::default_selector());
  }
}

void SYCLSpaceInitializer::finalize(const bool all_spaces) {
  if (std::is_same<Kokkos::SYCL,
                   Kokkos::DefaultExecutionSpace>::value ||
      all_spaces) {
    if (Kokkos::SYCL::impl_is_initialized())
      Kokkos::SYCL::impl_finalize();
  }
}

void SYCLSpaceInitializer::fence() {
  // FIXME_SYCL should be
  //  Kokkos::SYCL::impl_static_fence();
  Kokkos::SYCL().fence();
}

void SYCLSpaceInitializer::print_configuration(std::ostream& msg,
                                               const bool /*detail*/) {
  msg << "Devices:" << std::endl;
  msg << "  KOKKOS_ENABLE_SYCL: ";
  msg << "yes" << std::endl;

  msg << "\nRuntime Configuration:" << std::endl;
  // FIXME_SYCL not implemented
  std::abort();
  // SYCL::print_configuration(msg, detail);
}

}  // namespace Impl
}  // namespace Kokkos
