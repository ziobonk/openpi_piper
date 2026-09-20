cmake_minimum_required(VERSION 3.20)
project(zarr2parquet VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- vcpkg ----
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain")
endif()

# ---- Dependencies (all available via vcpkg) ----
find_package(nlohmann_json REQUIRED)
find_package(libjpeg-turbo CONFIG REQUIRED)  # libjpeg-turbo (vcpkg: libjpeg-turbo)
find_package(Arrow CONFIG REQUIRED)     # Apache Arrow
find_package(Parquet CONFIG REQUIRED)   # Apache Parquet C++
find_package(OpenMP REQUIRED)
find_package(blosc REQUIRED)            # blosc (vcpkg: blosc)
find_package(Threads REQUIRED)
find_package(CLI11 REQUIRED)            # CLI11 (vcpkg: cli11)
find_package(ZLIB REQUIRED)             # for zlib decompression fallback

# ---- Sources ----
set(SOURCES
    src/main.cpp
    src/zarr.cpp
    src/convert.cpp
    src/thread_pool.cpp
)

add_executable(zarr2parquet ${SOURCES})

target_include_directories(zarr2parquet PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(zarr2parquet PRIVATE
    libjpeg-turbo::turbojpeg-static
    Arrow::arrow_static
    Parquet::parquet_static
    blosc_static
    CLI11::CLI11
    nlohmann_json::nlohmann_json
    ZLIB::ZLIB
    Threads::Threads
    OpenMP::OpenMP_CXX
)

# Arrow needs C++17 at minimum
target_compile_features(zarr2parquet PRIVATE cxx_std_20)

# ---- Install ----
install(TARGETS zarr2parquet RUNTIME DESTINATION bin)
