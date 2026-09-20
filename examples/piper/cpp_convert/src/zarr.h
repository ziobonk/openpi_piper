#pragma once
/// Minimal zarr v2 reader — supports blosc-compressed chunked arrays.
/// Reads metadata from .zarray JSON and chunk data from binary blobs.
///
/// Supported:
///   - zarr v2 format (directory-based)
///   - blosc, zlib, gzip, and uncompressed chunks
///   - numeric dtypes: float32, float64, int32, int64, uint8
///   - arbitrary dimension arrays (sliced by first dimension)

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace zarr {

namespace fs = std::filesystem;

enum class DType : uint8_t {
    kFloat32,
    kFloat64,
    kInt32,
    kInt64,
    kUInt8,
};

struct ArrayMetadata {
    std::string path;       // relative path from zarr root
    std::vector<int64_t> shape;
    std::vector<int64_t> chunks;
    DType dtype;
    std::string compressor; // "blosc", "zlib", "gzip", or empty for raw
    int blosc_cname = 0;    // blosc compressor (0=lz4, 1=snappy, 2=zlib, 3=zstd)
    int blosc_clevel = 5;
    int blosc_shuffle = 1;  // 0=none, 1=byte, 2=bit
    size_t element_size = 0;
    char byte_order = '<';  // '<' little, '>' big
};

class ArrayReader {
public:
    /// Open a zarr array at `path` (the directory containing .zarray).
    static std::unique_ptr<ArrayReader> open(const fs::path& array_dir);

    /// Read a contiguous slice [start, end) along dim 0.
    /// `out` must be pre-allocated with size (end-start) * elem_count * element_size.
    void read_slice(int64_t start, int64_t end, void* out) const;

    /// Read a single element: index along dim 0.
    void read_element(int64_t index, void* out) const;

    const ArrayMetadata& meta() const { return _meta; }
    int64_t num_rows() const { return _meta.shape[0]; }
    size_t row_size_bytes() const;

private:
    ArrayMetadata _meta;
    fs::path _dir;
    std::vector<int64_t> _chunk_offsets;  // cumulative row offsets of chunks

    void _read_chunk(int64_t chunk_idx, std::vector<uint8_t>& buf) const;
    void _decompress_chunk(const std::vector<uint8_t>& compressed, void* out, size_t expected_size) const;
};


/// Open a zarr group and enumerate all arrays under data/.
struct DatasetArrays {
    ArrayMetadata left_joint;
    ArrayMetadata right_joint;
    ArrayMetadata left_gripper;
    ArrayMetadata right_gripper;
    ArrayMetadata timestamp;
    ArrayMetadata img0;
    ArrayMetadata img1;
    ArrayMetadata img2;
    // Optional
    std::optional<ArrayMetadata> action;
    std::vector<int64_t> episode_ends;  // cumulative frame counts
    int64_t total_frames = 0;
    int num_episodes = 0;
};

DatasetArrays open_dataset(const fs::path& zarr_root);

}  // namespace zarr
