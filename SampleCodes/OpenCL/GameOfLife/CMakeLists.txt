cmake_minimum_required(VERSION 3.20)

project(gameoflife LANGUAGES CXX)

find_package(Threads REQUIRED)

# Find/Fetch OpenCL
find_package(OpenCL
  # MODULE # CONFIG
  # QUIET
  #COMPONENTS
  #  ${REQUIRED_OPENCL_COMPONENTS}
)
if(NOT TARGET OpenCL::OpenCL)
  message(STATUS "OpenCL-SDK not found. Fetching...")
  FetchContent_Declare(
    OpenCL-SDK
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-SDK.git
    GIT_TAG        ff44fc38e298ff883716c4cc67a9976eb5205570 # v2022.05.18
    GIT_SUBMODULES
      external/OpenCL-ICD-Loader
      external/OpenCL-Headers
      external/OpenCL-CLHPP
  )

  set(OPENCL_SDK_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(OpenCL-SDK)
else()
  find_package(OpenCL
  # CONFIG
  REQUIRED
  # COMPONENTS
  #   ${RQUIRED_OPENCL_COMPONENTS}
  )
endif()

find_package(raylib 4.5.0 QUIET)
if (NOT raylib_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        raylib
        URL https://github.com/raysan5/raylib/archive/refs/tags/4.5.0.tar.gz
    )
    FetchContent_GetProperties(raylib)
    if (NOT raylib_POPULATED)
        set(FETCHCONTENT_QUIET NO)
        FetchContent_Populate(raylib)
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
        add_subdirectory(${raylib_SOURCE_DIR} ${raylib_BINARY_DIR})
    endif()
endif()

add_executable(gameoflife main.cpp)
target_compile_features(gameoflife PRIVATE cxx_std_17)
set_target_properties(gameoflife PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(gameoflife PRIVATE OpenCL::OpenCL Threads::Threads raylib)

get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(COPY_TO "${CMAKE_CURRENT_BINARY_DIR}/$<$<BOOL:${GENERATOR_IS_MULTI_CONFIG}>:$<CONFIG>>")

add_custom_command(
 OUTPUT "${COPY_TO}/step.cl"
 COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/step.cl" "${COPY_TO}/step.cl"
 COMMENT "Copying OpenCL kernel file to executable directory"
 DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/step.cl")

add_custom_target(gameoflife-kernel DEPENDS "${COPY_TO}/step.cl")
add_dependencies(gameoflife gameoflife-kernel)
