#ifndef RTL_SDR_DRIVER_H
#define RTL_SDR_DRIVER_H

#include <cstdint>
#include <complex>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <optional>

extern "C" {
#include <rtl-sdr.h>
}

/**
 * @brief RTL-SDR Driver class using librtlsdr
 * 
 * Provides RAII-based interface for RTL-SDR device control with:
 * - Device opening/closing
 * - Frequency, gain, and sample rate configuration
 * - Asynchronous IQ streaming with ring buffer storage
 * - Thread-safe operations
 */
class RtlSdrDriver {
public:
    using IQSample = std::complex<float>;
    using IQBuffer = std::vector<IQSample>;

    /**
     * @brief Construct a new RtlSdrDriver object
     * @param device_index Device index (0 for first device)
     */
    explicit RtlSdrDriver(uint32_t device_index = 0);

    /**
     * @brief Destructor - automatically closes device and stops streaming
     */
    ~RtlSdrDriver();

    // Delete copy constructor and assignment operator
    RtlSdrDriver(const RtlSdrDriver&) = delete;
    RtlSdrDriver& operator=(const RtlSdrDriver&) = delete;

    // Move constructor and assignment operator
    RtlSdrDriver(RtlSdrDriver&& other) noexcept;
    RtlSdrDriver& operator=(RtlSdrDriver&& other) noexcept;

    /**
     * @brief Open the RTL-SDR device
     * @return true if successful, false otherwise
     */
    bool open();

    /**
     * @brief Close the RTL-SDR device
     */
    void close();

    /**
     * @brief Check if device is open
     * @return true if device is open
     */
    bool is_open() const noexcept;

    /**
     * @brief Set center frequency in Hz
     * @param frequency_hz Frequency in Hz
     * @return true if successful, false otherwise
     */
    bool set_frequency(uint32_t frequency_hz);

    /**
     * @brief Get current center frequency
     * @return Current frequency in Hz, or empty if not set
     */
    std::optional<uint32_t> get_frequency() const noexcept;

    /**
     * @brief Set tuner gain
     * @param gain_db Gain in tenths of dB (e.g., 100 = 10.0 dB)
     * @return true if successful, false otherwise
     */
    bool set_gain(int gain_db);

    /**
     * @brief Set automatic gain control
     * @param enabled Enable or disable AGC
     * @return true if successful, false otherwise
     */
    bool set_agc(bool enabled);

    /**
     * @brief Get current gain
     * @return Current gain in tenths of dB, or empty if not set
     */
    std::optional<int> get_gain() const noexcept;

    /**
     * @brief Set sample rate in Hz
     * @param sample_rate_hz Sample rate in Hz
     * @return true if successful, false otherwise
     */
    bool set_sample_rate(uint32_t sample_rate_hz);

    /**
     * @brief Get current sample rate
     * @return Current sample rate in Hz, or empty if not set
     */
    std::optional<uint32_t> get_sample_rate() const noexcept;

    /**
     * @brief Start streaming IQ samples
     * @param buffer_size Size of the ring buffer in samples
     * @return true if successful, false otherwise
     */
    bool start_streaming(size_t buffer_size = 1024 * 1024);

    /**
     * @brief Stop streaming IQ samples
     */
    void stop_streaming();

    /**
     * @brief Check if streaming is active
     * @return true if streaming
     */
    bool is_streaming() const noexcept;

    /**
     * @brief Read IQ samples from ring buffer
     * @param samples Output buffer for samples
     * @param max_samples Maximum number of samples to read
     * @return Number of samples actually read
     */
    size_t read_samples(IQSample* samples, size_t max_samples);

    /**
     * @brief Read IQ samples from ring buffer into vector
     * @param max_samples Maximum number of samples to read
     * @return Vector of IQ samples
     */
    IQBuffer read_samples(size_t max_samples);

    /**
     * @brief Get number of samples available in ring buffer
     * @return Number of available samples
     */
    size_t available_samples() const noexcept;

    /**
     * @brief Get last error message
     * @return Error message string
     */
    std::string get_last_error() const;

    /**
     * @brief Get device count
     * @return Number of available RTL-SDR devices
     */
    static uint32_t get_device_count();

    /**
     * @brief Get device name
     * @param device_index Device index
     * @return Device name string
     */
    static std::string get_device_name(uint32_t device_index);

private:
    // Device handle
    rtlsdr_dev_t* device_;
    uint32_t device_index_;
    std::atomic<bool> is_open_;
    std::atomic<bool> is_streaming_;

    // Configuration
    std::atomic<uint32_t> frequency_hz_;
    std::atomic<int> gain_db_;
    std::atomic<uint32_t> sample_rate_hz_;

    // Ring buffer for IQ samples
    mutable std::mutex buffer_mutex_;
    std::vector<uint8_t> raw_buffer_;  // Raw uint8_t buffer from librtlsdr
    std::vector<IQSample> iq_buffer_;  // Converted complex samples
    size_t write_pos_;
    size_t read_pos_;
    size_t buffer_size_;
    std::atomic<size_t> available_count_;

    // Streaming thread
    std::thread streaming_thread_;
    std::atomic<bool> stop_streaming_flag_;

    // Error handling
    mutable std::mutex error_mutex_;
    std::string last_error_;

    // Internal methods
    void set_error(const std::string& error) const;
    void streaming_worker();
    static void rtlsdr_callback(unsigned char* buf, uint32_t len, void* ctx);
    void process_samples(const uint8_t* buf, size_t len);
    void convert_to_complex(const uint8_t* buf, size_t len, IQSample* out);
};

#endif // RTL_SDR_DRIVER_H

