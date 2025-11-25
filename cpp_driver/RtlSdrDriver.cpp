#include "RtlSdrDriver.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

// Static callback wrapper for librtlsdr
void RtlSdrDriver::rtlsdr_callback(unsigned char* buf, uint32_t len, void* ctx) {
    auto* driver = static_cast<RtlSdrDriver*>(ctx);
    if (driver && driver->is_streaming_.load()) {
        driver->process_samples(buf, len);
    }
}

RtlSdrDriver::RtlSdrDriver(uint32_t device_index)
    : device_(nullptr)
    , device_index_(device_index)
    , is_open_(false)
    , is_streaming_(false)
    , frequency_hz_(0)
    , gain_db_(0)
    , sample_rate_hz_(0)
    , write_pos_(0)
    , read_pos_(0)
    , buffer_size_(0)
    , available_count_(0)
    , stop_streaming_flag_(false) {
}

RtlSdrDriver::~RtlSdrDriver() {
    stop_streaming();
    close();
}

RtlSdrDriver::RtlSdrDriver(RtlSdrDriver&& other) noexcept
    : device_(other.device_)
    , device_index_(other.device_index_)
    , is_open_(other.is_open_.load())
    , is_streaming_(other.is_streaming_.load())
    , frequency_hz_(other.frequency_hz_.load())
    , gain_db_(other.gain_db_.load())
    , sample_rate_hz_(other.sample_rate_hz_.load())
    , raw_buffer_(std::move(other.raw_buffer_))
    , iq_buffer_(std::move(other.iq_buffer_))
    , write_pos_(other.write_pos_)
    , read_pos_(other.read_pos_)
    , buffer_size_(other.buffer_size_)
    , available_count_(other.available_count_.load())
    , streaming_thread_(std::move(other.streaming_thread_))
    , stop_streaming_flag_(other.stop_streaming_flag_.load()) {
    other.device_ = nullptr;
    other.is_open_ = false;
    other.is_streaming_ = false;
}

RtlSdrDriver& RtlSdrDriver::operator=(RtlSdrDriver&& other) noexcept {
    if (this != &other) {
        stop_streaming();
        close();

        device_ = other.device_;
        device_index_ = other.device_index_;
        is_open_ = other.is_open_.load();
        is_streaming_ = other.is_streaming_.load();
        frequency_hz_ = other.frequency_hz_.load();
        gain_db_ = other.gain_db_.load();
        sample_rate_hz_ = other.sample_rate_hz_.load();
        raw_buffer_ = std::move(other.raw_buffer_);
        iq_buffer_ = std::move(other.iq_buffer_);
        write_pos_ = other.write_pos_;
        read_pos_ = other.read_pos_;
        buffer_size_ = other.buffer_size_;
        available_count_ = other.available_count_.load();
        streaming_thread_ = std::move(other.streaming_thread_);
        stop_streaming_flag_ = other.stop_streaming_flag_.load();

        other.device_ = nullptr;
        other.is_open_ = false;
        other.is_streaming_ = false;
    }
    return *this;
}

bool RtlSdrDriver::open() {
    if (is_open_.load()) {
        set_error("Device already open");
        return false;
    }

    int result = rtlsdr_open(&device_, device_index_);
    if (result < 0) {
        set_error("Failed to open RTL-SDR device: " + std::to_string(result));
        return false;
    }

    is_open_ = true;
    return true;
}

void RtlSdrDriver::close() {
    if (!is_open_.load()) {
        return;
    }

    stop_streaming();

    if (device_) {
        rtlsdr_close(device_);
        device_ = nullptr;
    }

    is_open_ = false;
}

bool RtlSdrDriver::is_open() const noexcept {
    return is_open_.load();
}

bool RtlSdrDriver::set_frequency(uint32_t frequency_hz) {
    if (!is_open_.load()) {
        set_error("Device not open");
        return false;
    }

    int result = rtlsdr_set_center_freq(device_, frequency_hz);
    if (result < 0) {
        set_error("Failed to set frequency: " + std::to_string(result));
        return false;
    }

    frequency_hz_ = frequency_hz;
    return true;
}

std::optional<uint32_t> RtlSdrDriver::get_frequency() const noexcept {
    if (!is_open_.load() || frequency_hz_.load() == 0) {
        return std::nullopt;
    }
    return frequency_hz_.load();
}

bool RtlSdrDriver::set_gain(int gain_db) {
    if (!is_open_.load()) {
        set_error("Device not open");
        return false;
    }

    int result = rtlsdr_set_tuner_gain(device_, gain_db);
    if (result < 0) {
        set_error("Failed to set gain: " + std::to_string(result));
        return false;
    }

    gain_db_ = gain_db;
    return true;
}

bool RtlSdrDriver::set_agc(bool enabled) {
    if (!is_open_.load()) {
        set_error("Device not open");
        return false;
    }

    int result = rtlsdr_set_tuner_gain_mode(device_, enabled ? 0 : 1);
    if (result < 0) {
        set_error("Failed to set AGC mode: " + std::to_string(result));
        return false;
    }

    return true;
}

std::optional<int> RtlSdrDriver::get_gain() const noexcept {
    if (!is_open_.load()) {
        return std::nullopt;
    }
    return gain_db_.load();
}

bool RtlSdrDriver::set_sample_rate(uint32_t sample_rate_hz) {
    if (!is_open_.load()) {
        set_error("Device not open");
        return false;
    }

    int result = rtlsdr_set_sample_rate(device_, sample_rate_hz);
    if (result < 0) {
        set_error("Failed to set sample rate: " + std::to_string(result));
        return false;
    }

    sample_rate_hz_ = sample_rate_hz;
    return true;
}

std::optional<uint32_t> RtlSdrDriver::get_sample_rate() const noexcept {
    if (!is_open_.load() || sample_rate_hz_.load() == 0) {
        return std::nullopt;
    }
    return sample_rate_hz_.load();
}

bool RtlSdrDriver::start_streaming(size_t buffer_size) {
    if (!is_open_.load()) {
        set_error("Device not open");
        return false;
    }

    if (is_streaming_.load()) {
        set_error("Streaming already active");
        return false;
    }

    // Initialize ring buffer
    buffer_size_ = buffer_size;
    iq_buffer_.resize(buffer_size_);
    raw_buffer_.clear();
    write_pos_ = 0;
    read_pos_ = 0;
    available_count_ = 0;
    stop_streaming_flag_ = false;

    // Start async reading
    is_streaming_ = true;
    streaming_thread_ = std::thread(&RtlSdrDriver::streaming_worker, this);

    return true;
}

void RtlSdrDriver::stop_streaming() {
    if (!is_streaming_.load()) {
        return;
    }

    stop_streaming_flag_ = true;
    is_streaming_ = false;

    // Cancel async read
    if (device_ && is_open_.load()) {
        rtlsdr_cancel_async(device_);
    }

    // Wait for streaming thread to finish
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }
}

bool RtlSdrDriver::is_streaming() const noexcept {
    return is_streaming_.load();
}

size_t RtlSdrDriver::read_samples(IQSample* samples, size_t max_samples) {
    if (!samples || max_samples == 0) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    size_t available = available_count_.load();
    size_t to_read = std::min(max_samples, available);
    
    if (to_read == 0) {
        return 0;
    }

    // Read from ring buffer
    for (size_t i = 0; i < to_read; ++i) {
        samples[i] = iq_buffer_[read_pos_];
        read_pos_ = (read_pos_ + 1) % buffer_size_;
    }

    available_count_ -= to_read;
    return to_read;
}

RtlSdrDriver::IQBuffer RtlSdrDriver::read_samples(size_t max_samples) {
    IQBuffer result(max_samples);
    size_t read = read_samples(result.data(), max_samples);
    result.resize(read);
    return result;
}

size_t RtlSdrDriver::available_samples() const noexcept {
    return available_count_.load();
}

std::string RtlSdrDriver::get_last_error() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

uint32_t RtlSdrDriver::get_device_count() {
    return rtlsdr_get_device_count();
}

std::string RtlSdrDriver::get_device_name(uint32_t device_index) {
    const char* name = rtlsdr_get_device_name(device_index);
    return name ? std::string(name) : std::string("Unknown");
}

void RtlSdrDriver::set_error(const std::string& error) const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = error;
}

void RtlSdrDriver::streaming_worker() {
    if (!device_) {
        return;
    }

    // Start async read with callback
    rtlsdr_read_async(device_, rtlsdr_callback, this, 0, 0);
}

void RtlSdrDriver::process_samples(const uint8_t* buf, size_t len) {
    if (!buf || len == 0 || stop_streaming_flag_.load()) {
        return;
    }

    // Convert uint8_t samples to complex<float>
    // librtlsdr provides interleaved I/Q samples as uint8_t
    // Each sample is 8-bit, range 0-255, center at 127.5
    // Convert to float: (sample - 127.5) / 127.5
    const size_t num_samples = len / 2;  // I and Q are interleaved
    
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    for (size_t i = 0; i < num_samples; ++i) {
        // Convert I and Q from uint8_t to float
        float I = (static_cast<float>(buf[i * 2]) - 127.5f) / 127.5f;
        float Q = (static_cast<float>(buf[i * 2 + 1]) - 127.5f) / 127.5f;
        
        IQSample sample(I, Q);
        
        // Write to ring buffer (overwrite old data if buffer is full)
        iq_buffer_[write_pos_] = sample;
        write_pos_ = (write_pos_ + 1) % buffer_size_;
        
        // Update available count
        size_t current_available = available_count_.load();
        if (current_available < buffer_size_) {
            available_count_++;
        } else {
            // Buffer is full, advance read position (drop oldest sample)
            read_pos_ = (read_pos_ + 1) % buffer_size_;
        }
    }
}

void RtlSdrDriver::convert_to_complex(const uint8_t* buf, size_t len, IQSample* out) {
    const size_t num_samples = len / 2;
    for (size_t i = 0; i < num_samples; ++i) {
        float I = (static_cast<float>(buf[i * 2]) - 127.5f) / 127.5f;
        float Q = (static_cast<float>(buf[i * 2 + 1]) - 127.5f) / 127.5f;
        out[i] = IQSample(I, Q);
    }
}

