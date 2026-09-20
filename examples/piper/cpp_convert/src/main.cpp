/// zarr2parquet — High-performance dual-piper zarr → LeRobot parquet converter.
///
/// Converts diffusion_policy_piper zarr replay buffers to LeRobot v2 parquet
/// datasets.  Uses libjpeg-turbo for fast JPEG encoding, Apache Arrow for
/// parquet I/O, and a thread pool for parallel episode processing.
///
/// Usage:
///   zarr2parquet --input /path/to/replay_buffer.zarr \
///                --output ./dual_piper_lerobot \
///                --mode joint [--fps 50] [--quality 90]

#include "convert.h"
#include <CLI/CLI.hpp>
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app{"zarr2parquet — zarr → LeRobot parquet converter"};

    ConvertConfig cfg;
    app.add_option("--input,-i", cfg.zarr_path, "Input zarr directory")->required();
    app.add_option("--output,-o", cfg.output_dir, "Output LeRobot dataset directory")->required();
    app.add_option("--mode,-m", cfg.mode, "Control mode: 'joint' or 'eef'");
    app.add_option("--task", cfg.task_text, "Task description text");
    app.add_option("--fps", cfg.fps, "Frames per second (default: 50)");
    app.add_option("--quality,-q", cfg.jpeg_quality, "JPEG quality 1-100 (default: 90)");
    app.add_option("--threads,-t", cfg.num_threads,
                   "Number of threads (default: hardware concurrency)");
    app.add_option("--parallel,-p", cfg.parallel_episodes,
                   "Max parallel episodes (default: threads/2)");

    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(cfg.zarr_path)) {
        std::cerr << "[ERROR] zarr path not found: " << cfg.zarr_path << "\n";
        return 1;
    }

    std::cout << "[INFO] mode: " << cfg.mode << std::endl;
    std::cout << "[INFO] input: " << cfg.zarr_path << std::endl;
    std::cout << "[INFO] output: " << cfg.output_dir << std::endl;

    return run_converter(cfg);
}
