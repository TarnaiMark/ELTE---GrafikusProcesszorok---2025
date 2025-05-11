cmake_minimum_required (VERSION 3.20)
project(sobel LANGUAGES CXX)
find_package(OpenCL REQUIRED)
add_executable(sobel SobelFilter.cpp)
target_link_libraries(sobel OpenCL::OpenCL)

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
 OUTPUT "${COPY_TO}/sobel.cl"
 COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../sobel.cl" "${COPY_TO}/sobel.cl"
 COMMENT "Copying OpenCL kernel file to executable directory"
 DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../sobel.cl")

# We also have an image that we need to copy similarly:
add_custom_command(
 OUTPUT "${COPY_TO}/Valve_original.png"
 COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../Valve_original.png" "${COPY_TO}/Valve_original.png"
 COMMENT "Copying input image file to executable directory"
 DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../Valve_original.png")

# 4) We add a target, whose build can be triggered, and it requires the above commands to complete, producing their OUTPUTs:
add_custom_target(sobel-kernel-and-image DEPENDS "${COPY_TO}/sobel.cl" "${COPY_TO}/Valve_original.png")

# 5) We add a dependency between the main target and the kernel target,
# such that the main project will rebuild if we modify the kernel and that triggers the build of the kernel target:
add_dependencies(sobel sobel-kernel-and-image)
