cmake_minimum_required(VERSION 3.26)
project(Laplacian_Filter)

# add dependencies
include(cmake/CPM.cmake)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 20)

ADD_EXECUTABLE(${PROJECT_NAME} main.cpp)

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
TARGET_LINK_LIBRARIES (${PROJECT_NAME} ${OpenCV_LIBS})

find_package(spdlog CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} spdlog::spdlog spdlog::spdlog_header_only)

find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/res
        ${CMAKE_CURRENT_BINARY_DIR}/res)
