cmake_minimum_required(VERSION 3.18)
project(xfeat_lightglue LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS
  core imgproc imgcodecs highgui features2d calib3d)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

# LibTorch 2.2 still invokes CMake's legacy FindCUDA module internally. Bridge
# the modern CUDAToolkit result so both discovery paths use the same toolkit.
get_filename_component(_cuda_toolkit_root "${CUDAToolkit_BIN_DIR}" DIRECTORY)
set(CUDA_TOOLKIT_ROOT_DIR "${_cuda_toolkit_root}" CACHE PATH
    "CUDA root used by LibTorch's legacy FindCUDA" FORCE)
set(CUDA_NVCC_EXECUTABLE "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH
    "CUDA compiler used by LibTorch's legacy FindCUDA" FORCE)
set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH
    "CUDA compiler used when LibTorch enables the CUDA language" FORCE)

# Prefer the system-wide LibTorch installation on this machine while still
# allowing callers to override it with -DTorch_DIR or -DCMAKE_PREFIX_PATH.
if(NOT Torch_DIR AND EXISTS "/opt/libtorch/share/cmake/Torch/TorchConfig.cmake")
  list(PREPEND CMAKE_PREFIX_PATH "/opt/libtorch")
endif()
find_package(Torch CONFIG REQUIRED)

# This project is also consumed with add_subdirectory() by stereo_vo. Promote
# imported CUDA, yaml-cpp and LibTorch targets so their transitive usage
# requirements remain visible to targets created in the parent directory.
get_property(_xfeat_imported_targets DIRECTORY PROPERTY IMPORTED_TARGETS)
foreach(_target IN LISTS _xfeat_imported_targets)
  set_property(TARGET "${_target}" PROPERTY IMPORTED_GLOBAL TRUE)
endforeach()

# TensorRT's Debian packages do not provide a CMake config file. Locate the
# headers and runtime libraries without baking in a particular release.
set(TENSORRT_ROOT "" CACHE PATH "Optional TensorRT installation prefix")
set(_tensorrt_include_hints
  "${TENSORRT_ROOT}/include"
  /usr/include
  /usr/include/x86_64-linux-gnu
  /usr/local/include)
set(_tensorrt_library_hints
  "${TENSORRT_ROOT}/lib"
  "${TENSORRT_ROOT}/lib64"
  /usr/lib/x86_64-linux-gnu
  /usr/local/lib
  /usr/local/lib64)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${_tensorrt_include_hints})
find_library(TENSORRT_NVINFER_LIBRARY nvinfer HINTS ${_tensorrt_library_hints})
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin HINTS ${_tensorrt_library_hints})

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY OR
   NOT TENSORRT_PLUGIN_LIBRARY)
  message(FATAL_ERROR
    "TensorRT development files were not found. Install libnvinfer-dev and "
    "libnvinfer-plugin-dev, or configure with -DTENSORRT_ROOT=<prefix>.")
endif()

set(TENSORRT_VERSION "unknown")
if(EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_major_line
       REGEX "^#define NV_TENSORRT_MAJOR +[0-9]+")
  file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_minor_line
       REGEX "^#define NV_TENSORRT_MINOR +[0-9]+")
  string(REGEX MATCH "[0-9]+" TENSORRT_VERSION_MAJOR "${_trt_major_line}")
  string(REGEX MATCH "[0-9]+" TENSORRT_VERSION_MINOR "${_trt_minor_line}")
  if(TENSORRT_VERSION_MAJOR)
    set(TENSORRT_VERSION "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}")
  endif()
endif()

add_library(TensorRT::nvinfer UNKNOWN IMPORTED GLOBAL)
set_target_properties(TensorRT::nvinfer PROPERTIES
  IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}")

add_library(TensorRT::plugin UNKNOWN IMPORTED GLOBAL)
set_target_properties(TensorRT::plugin PROPERTIES
  IMPORTED_LOCATION "${TENSORRT_PLUGIN_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}")

set(XFEAT_COMMON_LIBS
  CUDA::cudart
  TensorRT::nvinfer
  TensorRT::plugin
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  yaml-cpp)

add_library(xfeat_libs SHARED src/xfeat.cpp)
target_include_directories(xfeat_libs PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(xfeat_libs PUBLIC ${XFEAT_COMMON_LIBS})
target_compile_options(xfeat_libs PRIVATE -O3 -Wall -Wextra)

add_library(lightglue_libs SHARED src/lightglue.cpp)
target_include_directories(lightglue_libs PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(lightglue_libs PUBLIC ${XFEAT_COMMON_LIBS})
target_compile_options(lightglue_libs PRIVATE -O3 -Wall -Wextra)

add_executable(match_test src/main.cpp)
target_link_libraries(match_test PRIVATE xfeat_libs lightglue_libs)

# Torch may require an ABI flag and other compile options supplied as a string.
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(xfeat_libs PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_compile_options(lightglue_libs PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_compile_options(match_test PRIVATE ${TORCH_CXX_FLAGS_LIST})

message(STATUS "OpenCV:  ${OpenCV_VERSION}")
message(STATUS "CUDA:    ${CUDAToolkit_VERSION} (${CUDAToolkit_ROOT})")
message(STATUS "LibTorch: ${Torch_VERSION} (${Torch_DIR})")
message(STATUS "TensorRT: ${TENSORRT_VERSION} (${TENSORRT_INCLUDE_DIR})")
