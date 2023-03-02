cmake_minimum_required(VERSION 3.15)
project(evaluate_depth)

find_package(PNG REQUIRED)
find_path(PNGPP_INCLUDE_DIRS "png++/png.hpp" REQUIRED)
# message(${PNGPP_INCLUDE_DIRS})

set(PROJECT_SOURCES
evaluate_depth.cpp
io_depth.h
log_colormap.h
utils.h
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${PNG_LIBRARY})
target_include_directories(${PROJECT_NAME} PRIVATE ${PNGPP_INCLUDE_DIRS} ${PNG_INCLUDE_DIR})
