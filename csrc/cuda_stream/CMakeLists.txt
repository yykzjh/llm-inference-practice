# 可选：全局编译选项控制 per-thread stream
option(USE_PER_THREAD_STREAM "Use --default-stream per-thread" OFF)

add_executable(default_stream default_stream.cu)
add_executable(blocking_stream blocking_stream.cu)
add_executable(synchronize_stream synchronize_stream.cu)

# 链接必要的库
target_link_libraries(default_stream PRIVATE cudart)
target_link_libraries(blocking_stream PRIVATE cudart)
target_link_libraries(synchronize_stream PRIVATE cudart)

# 如果需要 pthread（Linux）
if(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(default_stream PRIVATE Threads::Threads)
    target_link_libraries(blocking_stream PRIVATE Threads::Threads)
    target_link_libraries(synchronize_stream PRIVATE Threads::Threads)
endif()

if(USE_PER_THREAD_STREAM)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
    target_compile_definitions(default_stream PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(blocking_stream PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(synchronize_stream PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
    message(STATUS "Compiling with --default-stream per-thread")
else()
    message(STATUS "Compiling with legacy default stream")
endif()

# 输出构建说明
message("")
message("Build commands:")
message("  Legacy mode:      cmake -B build        && cmake --build build")
message("  Per-thread mode:  cmake -B build -DUSE_PER_THREAD_STREAM=ON && cmake --build build")
message("")
