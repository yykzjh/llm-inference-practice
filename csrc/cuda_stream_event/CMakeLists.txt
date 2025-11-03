add_executable(count_async_engine count_async_engine.cu)
add_executable(synchronize_stream_with_event synchronize_stream_with_event.cu)
add_executable(event_elapsed_time event_elapsed_time.cu)

target_link_libraries(count_async_engine PRIVATE cudart)
target_link_libraries(synchronize_stream_with_event PRIVATE cudart)
target_link_libraries(event_elapsed_time PRIVATE cudart)

if(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(event_elapsed_time PRIVATE Threads::Threads)
endif()
