cmake_minimum_required (VERSION 3.20)
project(histogram LANGUAGES CXX)
find_package(OpenCL REQUIRED)
add_executable(histogram Histogram.cpp)
target_link_libraries(histogram OpenCL::OpenCL)

# CMake magic related to copying the kernel file next to the binary:
# 1) Get a boolean representing if this is a multiple config generator:
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# 2) Create the path variable that points to the output directory, where the executable will be placed,
# and if this is a multi-config case, add to it the sub-directory with the config's name:
set(COPY_TO "${CMAKE_CURRENT_BINARY_DIR}/$<$<BOOL:${GENERATOR_IS_MULTI_CONFIG}>:$<CONFIG>>")

# 3) Now, we can specify the custom command, that will perform the copy
# We assume, the original kernel file is located one directory above the source folder (shared between the C/C++ versions)
# DEPENDS names the input trigger (e.g. modification of the kernel file) to this command:
add_custom_command(
 OUTPUT "${COPY_TO}/histogram.cl"
 COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../histogram.cl" "${COPY_TO}/histogram.cl"
 COMMENT "Copying OpenCL kernel file to executable directory"
 DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../histogram.cl")

# We also have an image that we need to copy similarly:
add_custom_command(
 OUTPUT "${COPY_TO}/NZ.jpg"
 COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../NZ.jpg" "${COPY_TO}/NZ.jpg"
 COMMENT "Copying input image file to executable directory"
 DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../NZ.jpg")

# 4) We add a target, whose build can be triggered, and it requires the above commands to complete, producing their OUTPUTs:
add_custom_target(histogram-kernel-and-image DEPENDS "${COPY_TO}/histogram.cl" "${COPY_TO}/NZ.jpg")

# 5) We add a dependency between the main target and the kernel target,
# such that the main project will rebuild if we modify the kernel and that triggers the build of the kernel target:
add_dependencies(histogram histogram-kernel-and-image)