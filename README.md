# SDR

Software-Defined Radio project with modular architecture for hardware control, signal processing, and user interface.

## Architecture

This project follows a modular design with clear separation of concerns:

### C++ Driver (`/cpp_driver`)
- **Purpose**: Handles low-level hardware access
- **Technology**: Uses `librtlsdr` for RTL-SDR device communication
- **Responsibilities**: Device initialization, configuration (frequency, gain, sample rate), and raw IQ data streaming

### Python Bindings (`/bindings`)
- **Purpose**: Bridges C++ driver to Python ecosystem
- **Technology**: Pybind11 for creating Python bindings
- **Responsibilities**: Exposes C++ driver functionality as a Python module, enabling seamless integration with Python DSP and UI components

### Python Modules (`/python`)
- **Acquisition** (`/python/acquisition`): High-level data acquisition and streaming interfaces
- **DSP** (`/python/dsp`): Digital signal processing algorithms and transformations
- **UI** (`/python/ui`): User interface components for visualization and control

## Project Structure

```
/SDR
  /cpp_driver          # C++ hardware driver using librtlsdr
  /python
    /acquisition       # Data acquisition module
    /dsp              # Digital signal processing
    /ui               # User interface
  /bindings           # Pybind11 bindings for C++ driver
```

## Design Principles

- **Modular Layout**: Clean separation between hardware access, bindings, and application logic
- **Language Separation**: C++ for performance-critical hardware operations, Python for rapid development of DSP and UI
- **Extensibility**: Each module can be developed and tested independently