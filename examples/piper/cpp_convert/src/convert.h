#pragma once
/// Episode converter: reads zarr slices, JPEG-encodes images, writes parquet.

#include "zarr.h"
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct ConvertConfig {
    std::string zarr_path;
    std::string output_dir;
    std::string mode = "joint";     // "joint" or "eef"
    std::string task_text = "pick up the block";
    int fps = 50;
    int jpeg_quality = 90;
    int num_threads = 0;            // 0 = auto
    int parallel_episodes = 0;      // 0 = auto
};

struct EpisodeStats {
    int episode_index;
    int length;
    std::vector<float> state_mean, state_std, state_min, state_max;
    std::vector<float> action_mean, action_std, action_min, action_max;
    struct ImageStats {
        float min[3], max[3], mean[3], std[3];
    };
    ImageStats img0, img1, img2;
};

/// Process a single episode: read → encode → write parquet.
EpisodeStats process_episode(
    const zarr::DatasetArrays& ds,
    const ConvertConfig& cfg,
    int ep_idx, int64_t ep_start, int64_t ep_end);

/// Write LeRobot v2 metadata files (info.json, episodes.jsonl, stats, etc.)
void write_metadata(
    const fs::path& output_dir,
    const zarr::DatasetArrays& ds,
    const ConvertConfig& cfg,
    const std::vector<EpisodeStats>& all_stats);

/// Main entry point for the converter.
int run_converter(const ConvertConfig& cfg);
