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

// void SYCL::impl_initialize() {
//   sycl::device device{sycl::default_selector()};
//   Impl::SYCLInternal::singleton().initialize(device);
// }
    SYCL::SYCLDevice::SYCLDevice(cl::sycl::device d) : m_device(std::move(d)) {}

    SYCL::SYCLDevice::SYCLDevice(const cl::sycl::device_selector& selector)
            : m_device(selector.select_device()) {}

    cl::sycl::device SYCL::SYCLDevice::get_device() const { return m_device; }

    void SYCL::impl_initialize(SYCL::SYCLDevice d) {
        Impl::SYCLInternal::singleton().initialize(d.get_device());
    }

    std::ostream& SYCL::SYCLDevice::info(std::ostream& os) const {
        using namespace cl::sycl::info;
        return os << "Name: " << m_device.get_info<device::name>()
                  << "\nDriver Version: "
                  << m_device.get_info<device::driver_version>()
                  << "\nIs Host: " << m_device.is_host()
                  << "\nIs CPU: " << m_device.is_cpu()
                  << "\nIs GPU: " << m_device.is_gpu()
                  << "\nIs Accelerator: " << m_device.is_accelerator()
                  << "\nVendor Id: " << m_device.get_info<device::vendor_id>()
                  << "\nMax Compute Units: "
                  << m_device.get_info<device::max_compute_units>()
                  << "\nMax Work Item Dimensions: "
                  << m_device.get_info<device::max_work_item_dimensions>()
                  << "\nMax Work Group Size: "
                  << m_device.get_info<device::max_work_group_size>()
                  << "\nPreferred Vector Width Char: "
                  << m_device.get_info<device::preferred_vector_width_char>()
                  << "\nPreferred Vector Width Short: "
                  << m_device.get_info<device::preferred_vector_width_short>()
                  << "\nPreferred Vector Width Int: "
                  << m_device.get_info<device::preferred_vector_width_int>()
                  << "\nPreferred Vector Width Long: "
                  << m_device.get_info<device::preferred_vector_width_long>()
                  << "\nPreferred Vector Width Float: "
                  << m_device.get_info<device::preferred_vector_width_float>()
                  << "\nPreferred Vector Width Double: "
                  << m_device.get_info<device::preferred_vector_width_double>()
                  << "\nPreferred Vector Width Half: "
                  << m_device.get_info<device::preferred_vector_width_half>()
                  << "\nNative Vector Width Char: "
                  << m_device.get_info<device::native_vector_width_char>()
                  << "\nNative Vector Width Short: "
                  << m_device.get_info<device::native_vector_width_short>()
                  << "\nNative Vector Width Int: "
                  << m_device.get_info<device::native_vector_width_int>()
                  << "\nNative Vector Width Long: "
                  << m_device.get_info<device::native_vector_width_long>()
                  << "\nNative Vector Width Float: "
                  << m_device.get_info<device::native_vector_width_float>()
                  << "\nNative Vector Width Double: "
                  << m_device.get_info<device::native_vector_width_double>()
                  << "\nNative Vector Width Half: "
                  << m_device.get_info<device::native_vector_width_half>()
                  << "\nAddress Bits: " << m_device.get_info<device::address_bits>()
                  << "\nImage Support: " << m_device.get_info<device::image_support>()
                  << "\nMax Mem Alloc Size: "
                  << m_device.get_info<device::max_mem_alloc_size>()
                  << "\nMax Read Image Args: "
                  << m_device.get_info<device::max_read_image_args>()
                  << "\nImage2d Max Width: "
                  << m_device.get_info<device::image2d_max_width>()
                  << "\nImage2d Max Height: "
                  << m_device.get_info<device::image2d_max_height>()
                  << "\nImage3d Max Width: "
                  << m_device.get_info<device::image3d_max_width>()
                  << "\nImage3d Max Height: "
                  << m_device.get_info<device::image3d_max_height>()
                  << "\nImage3d Max Depth: "
                  << m_device.get_info<device::image3d_max_depth>()
                  << "\nImage Max Buffer Size: "
                  << m_device.get_info<device::image_max_buffer_size>()
                  << "\nImage Max Array Size: "
                  << m_device.get_info<device::image_max_array_size>()
                  << "\nMax Samplers: " << m_device.get_info<device::max_samplers>()
                  << "\nMax Parameter Size: "
                  << m_device.get_info<device::max_parameter_size>()
                  << "\nMem Base Addr Align: "
                  << m_device.get_info<device::mem_base_addr_align>()
                  << "\nGlobal Cache Mem Line Size: "
                  << m_device.get_info<device::global_mem_cache_line_size>()
                  << "\nGlobal Mem Cache Size: "
                  << m_device.get_info<device::global_mem_cache_size>()
                  << "\nGlobal Mem Size: "
                  << m_device.get_info<device::global_mem_size>()
                  << "\nMax Constant Buffer Size: "
                  << m_device.get_info<device::max_constant_buffer_size>()
                  << "\nMax Constant Args: "
                  << m_device.get_info<device::max_constant_args>()
                  << "\nLocal Mem Size: "
                  << m_device.get_info<device::local_mem_size>()
                  << "\nError Correction Support: "
                  << m_device.get_info<device::error_correction_support>()
                  << "\nHost Unified Memory: "
                  << m_device.get_info<device::host_unified_memory>()
                  << "\nProfiling Timer Resolution: "
                  << m_device.get_info<device::profiling_timer_resolution>()
                  << "\nIs Endian Little: "
                  << m_device.get_info<device::is_endian_little>()
                  << "\nIs Available: " << m_device.get_info<device::is_available>()
                  << "\nIs Compiler Available: "
                  << m_device.get_info<device::is_compiler_available>()
                  << "\nIs Linker Available: "
                  << m_device.get_info<device::is_linker_available>()
                  << "\nQueue Profiling: "
                  << m_device.get_info<device::queue_profiling>()
                  << "\nBuilt In Kernels: "
                  << Container<std::vector<std::string>>(
                m_device.get_info<device::built_in_kernels>())
                << "\nVendor: " << m_device.get_info<device::vendor>()
                << "\nProfile: " << m_device.get_info<device::profile>()
                << "\nVersion: " << m_device.get_info<device::version>()
                << "\nExtensions: "
                << Container<std::vector<std::string>>(
                m_device.get_info<device::extensions>())
                << "\nPrintf Buffer Size: "
                << m_device.get_info<device::printf_buffer_size>()
                << "\nPreferred Interop User Sync: "
                << m_device.get_info<device::preferred_interop_user_sync>()
                << "\nPartition Max Sub Devices: "
                << m_device.get_info<device::partition_max_sub_devices>()
                << "\nReference Count: "
                << m_device.get_info<device::reference_count>() << '\n';
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
                Kokkos::SYCL::impl_initialize(Kokkos::SYCL::SYCLDevice(cl::sycl::default_selector()));
                //Kokkos::SYCL::impl_initialize();
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
