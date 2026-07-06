cmake_minimum_required(VERSION 3.18)
project(libtorch_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Point CMake to the extracted LibTorch directory.
# Pass at configure time:  cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)

add_executable(inference inference.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")

# LibTorch ships as shared libs — copy them next to the binary on Windows.
# On Linux this is handled by LD_LIBRARY_PATH or rpath, but the block is
# harmless to keep for cross-platform portability.
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${TORCH_DLLS} $<TARGET_FILE_DIR:inference>)
endif()
