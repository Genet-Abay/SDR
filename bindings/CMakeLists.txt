cmake_minimum_required(VERSION 3.15)
project(sdrhw)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find librtlsdr
# On Windows, we might need to specify the path manually
# On Linux, pkg-config is typically used
if(WIN32)
    # Windows: Try to find librtlsdr in common locations
    find_path(RTLSDR_INCLUDE_DIR
        NAMES rtl-sdr.h
        PATHS
            ${CMAKE_SOURCE_DIR}/third_party/librtlsdr/include
            ${CMAKE_SOURCE_DIR}/../third_party/librtlsdr/include
            C:/librtlsdr/include
            C:/rtl-sdr/include
            ENV RTLSDR_INCLUDE_DIR
    )
    
    find_library(RTLSDR_LIBRARY
        NAMES rtlsdr
        PATHS
            ${CMAKE_SOURCE_DIR}/third_party/librtlsdr/lib
            ${CMAKE_SOURCE_DIR}/../third_party/librtlsdr/lib
            C:/librtlsdr/lib
            C:/rtl-sdr/lib
            ENV RTLSDR_LIB_DIR
    )
    
    if(NOT RTLSDR_INCLUDE_DIR OR NOT RTLSDR_LIBRARY)
        message(WARNING "librtlsdr not found. Please set RTLSDR_INCLUDE_DIR and RTLSDR_LIB_DIR environment variables or install librtlsdr.")
    endif()
else()
    # Linux: Use pkg-config
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(RTLSDR REQUIRED librtlsdr)
    
    set(RTLSDR_INCLUDE_DIR ${RTLSDR_INCLUDE_DIRS})
    set(RTLSDR_LIBRARY ${RTLSDR_LIBRARIES})
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_driver
    ${RTLSDR_INCLUDE_DIR}
)

# Source files
set(SOURCES
    sdrhw.cpp
    ../cpp_driver/RtlSdrDriver.cpp
)

# Create Python module
pybind11_add_module(sdrhw ${SOURCES})

# Link libraries
target_link_libraries(sdrhw PRIVATE ${RTLSDR_LIBRARY})

# Platform-specific settings
if(WIN32)
    # Windows: Set output extension to .pyd
    set_target_properties(sdrhw PROPERTIES
        SUFFIX ".pyd"
        PREFIX ""
    )
    
    # Copy DLL to output directory if needed
    if(RTLSDR_LIBRARY)
        get_filename_component(RTLSDR_DLL_DIR ${RTLSDR_LIBRARY} DIRECTORY)
        # Try to find the DLL
        file(GLOB RTLSDR_DLL "${RTLSDR_DLL_DIR}/*.dll")
        if(RTLSDR_DLL)
            add_custom_command(TARGET sdrhw POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${RTLSDR_DLL}
                $<TARGET_FILE_DIR:sdrhw>
            )
        endif()
    endif()
else()
    # Linux: Set RPATH for runtime library loading
    set_target_properties(sdrhw PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# Compiler-specific options
if(MSVC)
    # Disable some MSVC warnings
    target_compile_options(sdrhw PRIVATE /W4)
    # Disable specific warnings if needed
    # target_compile_definitions(sdrhw PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    # GCC/Clang options
    target_compile_options(sdrhw PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Installation (optional)
install(TARGETS sdrhw
    COMPONENT python
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

