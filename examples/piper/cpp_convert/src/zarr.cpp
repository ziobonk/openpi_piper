#include "zarr.h"
#include <blosc.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <zlib.h>

using json = nlohmann::json;

namespace zarr {

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

static DType _parse_dtype(const std::string& s) {
    if (s == "<f4" || s == "float32") return DType::kFloat32;
    if (s == "<f8" || s == "float64") return DType::kFloat64;
    if (s == "<i4" || s == "int32")   return DType::kInt32;
    if (s == "<i8" || s == "int64")   return DType::kInt64;
    if (s == "|u1" || s == "uint8")   return DType::kUInt8;
    throw std::runtime_error("Unsupported dtype: " + s);
}

static size_t _dtype_size(DType dt) {
    switch (dt) {
        case DType::kFloat32: return 4;
        case DType::kFloat64: return 8;
        case DType::kInt32:   return 4;
        case DType::kInt64:   return 8;
        case DType::kUInt8:   return 1;
    }
    return 0;
}

static int _blosc_cname(const std::string& s) {
    if (s == "lz4" || s.empty())    return 0;
    if (s == "snappy")              return 1;
    if (s == "zlib")                return 2;
    if (s == "zstd")                return 3;
    if (s == "blosclz")             return 0;
    return 0;
}

// ---------------------------------------------------------------------------
// ArrayReader
// ---------------------------------------------------------------------------

std::unique_ptr<ArrayReader> ArrayReader::open(const fs::path& array_dir) {
    auto r = std::make_unique<ArrayReader>();
    r->_dir = array_dir;

    // Read .zarray metadata
    auto meta_path = array_dir / ".zarray";
    std::ifstream f(meta_path);
    if (!f) throw std::runtime_error("Cannot open " + meta_path.string());
    json j = json::parse(f);

    r->_meta.path = array_dir.string();
    r->_meta.shape = j["shape"].get<std::vector<int64_t>>();
    r->_meta.chunks = j["chunks"].get<std::vector<int64_t>>();
    r->_meta.dtype = _parse_dtype(j["dtype"].get<std::string>());
    r->_meta.element_size = _dtype_size(r->_meta.dtype);

    if (j.contains("compressor") && !j["compressor"].is_null()) {
        auto& comp = j["compressor"];
        r->_meta.compressor = comp["id"].get<std::string>();
        if (r->_meta.compressor == "blosc") {
            r->_meta.blosc_cname = _blosc_cname(comp.value("cname", "lz4"));
            r->_meta.blosc_clevel = comp.value("clevel", 5);
            r->_meta.blosc_shuffle = comp.value("shuffle", 1);
        }
    } else {
        r->_meta.compressor = "raw";
    }

    // Byte order
    std::string dt_str = j["dtype"].get<std::string>();
    r->_meta.byte_order = (dt_str[0] == '>' || dt_str[0] == '|') ? dt_str[0] : '<';

    // Pre-compute chunk offsets (cumulative rows along dim 0)
    int64_t chunk_size = r->_meta.chunks[0];
    int64_t total = r->_meta.shape[0];
    for (int64_t off = 0; off < total; off += chunk_size) {
        r->_chunk_offsets.push_back(off);
    }

    return r;
}

size_t ArrayReader::row_size_bytes() const {
    size_t elems = 1;
    for (size_t d = 1; d < _meta.shape.size(); ++d) {
        elems *= static_cast<size_t>(_meta.shape[d]);
    }
    return elems * _meta.element_size;
}

void ArrayReader::read_slice(int64_t start, int64_t end, void* out) const {
    if (start < 0 || end > _meta.shape[0] || start >= end) {
        throw std::out_of_range("read_slice: invalid range");
    }

    int64_t chunk_size = _meta.chunks[0];
    size_t row_sz = row_size_bytes();
    auto* dst = static_cast<uint8_t*>(out);

    std::vector<uint8_t> chunk_buf;
    int64_t pos = start;
    while (pos < end) {
        int64_t chunk_idx = pos / chunk_size;
        int64_t chunk_start = chunk_idx * chunk_size;
        int64_t chunk_end = std::min(chunk_start + chunk_size, _meta.shape[0]);
        int64_t read_start = pos;
        int64_t read_end = std::min(end, chunk_end);
        size_t read_rows = static_cast<size_t>(read_end - read_start);

        _read_chunk(chunk_idx, chunk_buf);

        // Copy the relevant rows from the chunk
        int64_t row_offset_in_chunk = read_start - chunk_start;
        const auto* src = chunk_buf.data() + row_offset_in_chunk * row_sz;
        size_t copy_sz = read_rows * row_sz;
        std::memcpy(dst, src, copy_sz);
        dst += copy_sz;
        pos = read_end;
    }
}

void ArrayReader::read_element(int64_t index, void* out) const {
    read_slice(index, index + 1, out);
}

static std::string _chunk_filename(int64_t chunk_idx, const std::vector<int64_t>& chunks) {
    // Build chunk file name from chunk index along dim 0.
    // All other dims start at index 0 since we read sequentially.
    // 2D array → "0.0", 4D array → "0.0.0.0"
    std::string name = std::to_string(chunk_idx);
    for (size_t d = 1; d < chunks.size(); ++d) name += ".0";
    return name;
}

void ArrayReader::_read_chunk(int64_t chunk_idx, std::vector<uint8_t>& buf) const {
    // Chunk file name: e.g. "0.0.0.0" for 4D, "0.0" for 2D
    int64_t row_start = chunk_idx * _meta.chunks[0];
    auto chunk_path = _dir / _chunk_filename(chunk_idx, _meta.chunks);

    std::ifstream f(chunk_path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open chunk: " + chunk_path.string());

    size_t file_sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> compressed(file_sz);
    f.read(reinterpret_cast<char*>(compressed.data()), file_sz);

    // zarr pads last chunk to full chunk size before compression;
    // always decompress to full chunk size, then only use valid rows.
    size_t full_chunk_sz = static_cast<size_t>(_meta.chunks[0]) * row_size_bytes();

    if (_meta.compressor == "raw") {
        buf = std::move(compressed);
    } else {
        buf.resize(full_chunk_sz);
        _decompress_chunk(compressed, buf.data(), full_chunk_sz);
    }
}

void ArrayReader::_decompress_chunk(const std::vector<uint8_t>& compressed,
                                     void* out, size_t expected_size) const {
    if (_meta.compressor == "blosc") {
        int ret = blosc_decompress(compressed.data(), out, expected_size);
        if (ret <= 0) {
            throw std::runtime_error("blosc decompression failed: " + std::to_string(ret));
        }
    } else if (_meta.compressor == "zlib" || _meta.compressor == "gzip") {
        uLongf dest_len = expected_size;
        int ret = uncompress(static_cast<Bytef*>(out), &dest_len,
                             compressed.data(), compressed.size());
        if (ret != Z_OK) {
            throw std::runtime_error("zlib decompression failed: " + std::to_string(ret));
        }
    } else {
        throw std::runtime_error("Unknown compressor: " + _meta.compressor);
    }
}

// ---------------------------------------------------------------------------
// Dataset discovery
// ---------------------------------------------------------------------------

static ArrayMetadata _open_array(const fs::path& root, const std::string& rel) {
    auto reader = ArrayReader::open(root / rel);
    return reader->meta();
}

DatasetArrays open_dataset(const fs::path& zarr_root) {
    DatasetArrays ds;

    auto data_dir = zarr_root / "data";
    ds.left_joint     = _open_array(data_dir, "left_robot_joint");
    ds.right_joint    = _open_array(data_dir, "right_robot_joint");
    ds.left_gripper   = _open_array(data_dir, "left_gripper_angle");
    ds.right_gripper  = _open_array(data_dir, "right_gripper_angle");
    ds.timestamp      = _open_array(data_dir, "timestamp");

    // Images — try new naming convention first
    if (fs::exists(data_dir / "img_camera_0")) {
        ds.img0 = _open_array(data_dir, "img_camera_0");
        ds.img1 = _open_array(data_dir, "img_camera_1");
        ds.img2 = _open_array(data_dir, "img_camera_2");
    } else if (fs::exists(data_dir / "img")) {
        ds.img0 = _open_array(data_dir, "img");
        ds.img1 = ds.img0;  // stacked format: use same array with offset along dim1
        ds.img2 = ds.img0;
    }

    // Optional action array
    if (fs::exists(data_dir / "action")) {
        ds.action = _open_array(data_dir, "action");
    }

    // Read episode_ends from meta/
    auto ep_path = zarr_root / "meta" / "episode_ends";
    auto reader = ArrayReader::open(ep_path);
    ds.episode_ends.resize(reader->num_rows());
    reader->read_slice(0, reader->num_rows(), ds.episode_ends.data());

    ds.total_frames = ds.episode_ends.back();
    ds.num_episodes = static_cast<int>(ds.episode_ends.size());

    return ds;
}

}  // namespace zarr
