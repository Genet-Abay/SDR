#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <vector>
#include "../cpp_driver/RtlSdrDriver.h"

namespace py = pybind11;

// Helper function to convert IQBuffer to numpy array
py::array_t<std::complex<float>> read_samples_numpy(RtlSdrDriver& driver, size_t max_samples) {
    // Read samples from driver
    auto samples = driver.read_samples(max_samples);
    
    // Create numpy array with the correct size
    py::array_t<std::complex<float>> result(samples.size());
    
    // Copy data to numpy array
    auto buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < samples.size(); ++i) {
        buf(i) = samples[i];
    }
    
    return result;
}

PYBIND11_MODULE(sdrhw, m) {
    m.doc() = "RTL-SDR Hardware Driver Python Bindings";

    py::class_<RtlSdrDriver>(m, "RtlSdrDriver")
        .def(py::init<uint32_t>(), 
             py::arg("device_index") = 0,
             "Construct RtlSdrDriver\n"
             "\n"
             "Args:\n"
             "    device_index: Device index (0 for first device)")
        
        .def("open", &RtlSdrDriver::open,
             "Open the RTL-SDR device\n"
             "\n"
             "Returns:\n"
             "    bool: True if successful, False otherwise")
        
        .def("close", &RtlSdrDriver::close,
             "Close the RTL-SDR device")
        
        .def("set_freq", &RtlSdrDriver::set_frequency,
             py::arg("frequency_hz"),
             "Set center frequency in Hz\n"
             "\n"
             "Args:\n"
             "    frequency_hz: Frequency in Hz\n"
             "\n"
             "Returns:\n"
             "    bool: True if successful, False otherwise")
        
        .def("set_sample_rate", &RtlSdrDriver::set_sample_rate,
             py::arg("sample_rate_hz"),
             "Set sample rate in Hz\n"
             "\n"
             "Args:\n"
             "    sample_rate_hz: Sample rate in Hz\n"
             "\n"
             "Returns:\n"
             "    bool: True if successful, False otherwise")
        
        .def("read_samples", &read_samples_numpy,
             py::arg("max_samples"),
             "Read IQ samples from ring buffer\n"
             "\n"
             "Args:\n"
             "    max_samples: Maximum number of samples to read\n"
             "\n"
             "Returns:\n"
             "    numpy.ndarray: Array of complex64 samples")
        
        .def("is_open", &RtlSdrDriver::is_open,
             "Check if device is open\n"
             "\n"
             "Returns:\n"
             "    bool: True if device is open")
        
        .def("start_streaming", &RtlSdrDriver::start_streaming,
             py::arg("buffer_size") = 1024 * 1024,
             "Start streaming IQ samples\n"
             "\n"
             "Args:\n"
             "    buffer_size: Size of the ring buffer in samples (default: 1M)\n"
             "\n"
             "Returns:\n"
             "    bool: True if successful, False otherwise")
        
        .def("stop_streaming", &RtlSdrDriver::stop_streaming,
             "Stop streaming IQ samples")
        
        .def("is_streaming", &RtlSdrDriver::is_streaming,
             "Check if streaming is active\n"
             "\n"
             "Returns:\n"
             "    bool: True if streaming")
        
        .def("available_samples", &RtlSdrDriver::available_samples,
             "Get number of samples available in ring buffer\n"
             "\n"
             "Returns:\n"
             "    int: Number of available samples")
        
        .def("get_last_error", &RtlSdrDriver::get_last_error,
             "Get last error message\n"
             "\n"
             "Returns:\n"
             "    str: Error message string")
        
        .def_static("get_device_count", &RtlSdrDriver::get_device_count,
                    "Get number of available RTL-SDR devices\n"
                    "\n"
                    "Returns:\n"
                    "    int: Number of devices")
        
        .def_static("get_device_name", &RtlSdrDriver::get_device_name,
                    py::arg("device_index"),
                    "Get device name\n"
                    "\n"
                    "Args:\n"
                    "    device_index: Device index\n"
                    "\n"
                    "Returns:\n"
                    "    str: Device name");
}

