cmake_minimum_required(VERSION 3.20)
project(histogram LANGUAGES CXX CUDA)
enable_language(CUDA)
add_executable(histogram Histogram.cu)
target_link_libraries(histogram cuda)
set_target_properties(histogram PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# CMake magic related to copying the input file next to the binary:
# 1) Get a boolean representing if this is a multiple config generator:
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# 2) Create the path variable that points to the output directory, where the executable will be placed,
# and if this is a multi-config case, add to it the sub-directory with the config's name:
set(COPY_TO "${CMAKE_CURRENT_BINARY_DIR}/$<$<BOOL:${GENERATOR_IS_MULTI_CONFIG}>:$<CONFIG>>")

# 3) Now, we can specify the custom command, that will perform the copy
# We assume, the original input file is located in the source folder:
# DEPENDS names the input trigger (e.g. modification of the input file) to this command:

add_custom_command(
 OUTPUT "${COPY_TO}/NZ.jpg"
 COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/NZ.jpg" "${COPY_TO}/NZ.jpg"
 COMMENT "Copying input image file to executable directory"
 DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/NZ.jpg")

# 4) We add a target, whose build can be triggered, and it requires the above commands to complete, producing their OUTPUTs:
add_custom_target(histogram-image DEPENDS "${COPY_TO}/NZ.jpg")

# 5) We add a dependency between the main target and the image target,
# such that the main project will rebuild if we modify input image and that triggers the build of the image target:
add_dependencies(histogram histogram-image)