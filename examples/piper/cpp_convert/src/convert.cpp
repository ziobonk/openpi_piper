#include "convert.h"
#include "thread_pool.h"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <fstream>
#include <iostream>
#include <turbojpeg.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>

using json = nlohmann::json;

#define THROW_IF_NOT_OK(expr) do { auto _s = (expr); if (!_s.ok()) throw std::runtime_error(_s.ToString()); } while(0)

// ---------------------------------------------------------------------------
// JPEG encoding
// ---------------------------------------------------------------------------
static std::vector<uint8_t> encode_jpeg(const uint8_t* rgb, int h, int w, int quality) {
    tjhandle tj = tjInitCompress();
    unsigned char* buf = nullptr;
    unsigned long sz = 0;
    tjCompress2(tj, rgb, w, 0, h, TJPF_RGB, &buf, &sz, TJSAMP_420, quality, TJFLAG_FASTDCT);
    std::vector<uint8_t> r(buf, buf + sz);
    tjFree(buf);
    tjDestroy(tj);
    return r;
}

// ---------------------------------------------------------------------------
// Image stats
// ---------------------------------------------------------------------------
static void compute_image_stats(const uint8_t* data, int64_t n, int h, int w,
                                float min_out[3], float max_out[3],
                                float mean_out[3], float std_out[3]) {
    double min_v[3] = {255,255,255}, max_v[3] = {0,0,0}, sum[3] = {0}, sq[3] = {0};
    int64_t total = n * h * w;
    for (int64_t p = 0; p < total; ++p) {
        int c = p % 3;
        double v = data[p];
        if (v < min_v[c]) min_v[c] = v;
        if (v > max_v[c]) max_v[c] = v;
        sum[c] += v; sq[c] += v * v;
    }
    for (int c = 0; c < 3; ++c) {
        min_out[c] = (float)min_v[c]; max_out[c] = (float)max_v[c];
        mean_out[c] = (float)(sum[c] / total);
        float var = (float)(sq[c]/total - mean_out[c]*mean_out[c]);
        std_out[c] = std::sqrt(std::max(0.0f, var));
    }
}

// ---------------------------------------------------------------------------
// HuggingFace schema metadata
// ---------------------------------------------------------------------------
// The `datasets` library (used by LeRobot) reads the "huggingface" schema metadata to
// recognize image columns and decode them. Without it, load_dataset("parquet") returns
// images as raw {"bytes", "path"} structs and LeRobot crashes with
// "RuntimeError: Could not infer dtype of dict".
static std::shared_ptr<arrow::KeyValueMetadata> hf_schema_metadata() {
    json features;

    json value_f32 = {{"dtype", "float32"}, {"_type", "Value"}};
    json seq_state = {{"feature", value_f32}, {"length", 14}, {"_type", "Sequence"}};
    features["state"] = seq_state;
    features["actions"] = seq_state;

    json value_i64 = {{"dtype", "int64"}, {"_type", "Value"}};
    features["timestamp"] = value_f32;
    features["frame_index"] = value_i64;
    features["episode_index"] = value_i64;
    features["index"] = value_i64;
    features["task_index"] = value_i64;

    json image_feat = {{"_type", "Image"}};
    features["image"] = image_feat;
    features["wrist_image"] = image_feat;
    features["wrist_image_right"] = image_feat;

    json meta;
    meta["info"]["features"] = features;
    auto kv = std::make_shared<arrow::KeyValueMetadata>();
    kv->Append("huggingface", meta.dump());
    return kv;
}

// ---------------------------------------------------------------------------
// Per-episode processing
// ---------------------------------------------------------------------------
EpisodeStats process_episode(
    const zarr::DatasetArrays& ds, const ConvertConfig& cfg,
    int ep_idx, int64_t ep_start, int64_t ep_end)
{
    int64_t n = ep_end - ep_start;
    std::cout << "[INFO] Episode " << ep_idx << ": " << n
              << " frames (" << ep_start << ".." << ep_end-1 << ")\n";

    auto left_r  = zarr::ArrayReader::open(ds.left_joint.path);
    auto right_r = zarr::ArrayReader::open(ds.right_joint.path);
    auto lg_r    = zarr::ArrayReader::open(ds.left_gripper.path);
    auto rg_r    = zarr::ArrayReader::open(ds.right_gripper.path);
    auto img0_r  = zarr::ArrayReader::open(ds.img0.path);
    auto img1_r  = zarr::ArrayReader::open(ds.img1.path);
    auto img2_r  = zarr::ArrayReader::open(ds.img2.path);

    int64_t h0 = ds.img0.shape[1], w0 = ds.img0.shape[2];
    int64_t h1 = ds.img1.shape[1], w1 = ds.img1.shape[2];
    int64_t h2 = ds.img2.shape[1], w2 = ds.img2.shape[2];
    size_t jsz = 6 * sizeof(float);

    // zarr 数据是 float64，先读入 double 再转 float32
    std::vector<double> left(n*6), right(n*6), lg(n), rg(n);
    std::vector<uint8_t> img0(n*h0*w0*3), img1(n*h1*w1*3), img2(n*h2*w2*3);

    left_r->read_slice(ep_start, ep_end, left.data());
    right_r->read_slice(ep_start, ep_end, right.data());
    lg_r->read_slice(ep_start, ep_end, lg.data());
    rg_r->read_slice(ep_start, ep_end, rg.data());
    img0_r->read_slice(ep_start, ep_end, img0.data());
    img1_r->read_slice(ep_start, ep_end, img1.data());
    img2_r->read_slice(ep_start, ep_end, img2.data());

    // Build state & actions (14-dim float32)
    std::vector<float> state(n*14), actions(n*14);
    for (int64_t i = 0; i < n; ++i) {
        float* s = state.data() + i*14;
        float* a = actions.data() + i*14;
        for (int j=0; j<6; ++j) { s[j]=(float)left[i*6+j]; s[7+j]=(float)right[i*6+j]; }
        s[6]=(float)lg[i]; s[13]=(float)rg[i];
        std::memcpy(a, s, 14*sizeof(float));
    }

    // ---- Build Arrow table ----
    auto img_type = arrow::struct_({
        arrow::field("bytes", arrow::binary()),
        arrow::field("path", arrow::utf8())});

    // List<float32> builders
    auto lf_type = arrow::list(arrow::float32());
    arrow::ListBuilder state_b(arrow::default_memory_pool(),
                               std::make_shared<arrow::FloatBuilder>());
    arrow::ListBuilder action_b(arrow::default_memory_pool(),
                                std::make_shared<arrow::FloatBuilder>());
    arrow::FloatBuilder ts_b;
    arrow::Int64Builder fi_b, ei_b, ix_b, ti_b;

    // Struct builders — 每个独立创建，不能共享 field builders
    auto make_img_fields = []() {
        return std::vector<std::shared_ptr<arrow::ArrayBuilder>>{
            std::make_shared<arrow::BinaryBuilder>(),
            std::make_shared<arrow::StringBuilder>()};
    };
    arrow::StructBuilder img_b(img_type, arrow::default_memory_pool(), make_img_fields());
    arrow::StructBuilder wrist_b(img_type, arrow::default_memory_pool(), make_img_fields());
    arrow::StructBuilder wrist_r_b(img_type, arrow::default_memory_pool(), make_img_fields());

    auto* bytes_b   = static_cast<arrow::BinaryBuilder*>(img_b.field_builder(0));
    auto* path_b    = static_cast<arrow::StringBuilder*>(img_b.field_builder(1));
    auto* wbytes_b  = static_cast<arrow::BinaryBuilder*>(wrist_b.field_builder(0));
    auto* wpath_b   = static_cast<arrow::StringBuilder*>(wrist_b.field_builder(1));
    auto* rbytes_b  = static_cast<arrow::BinaryBuilder*>(wrist_r_b.field_builder(0));
    auto* rpath_b   = static_cast<arrow::StringBuilder*>(wrist_r_b.field_builder(1));

    auto* sf_builder = static_cast<arrow::FloatBuilder*>(state_b.value_builder());
    auto* af_builder = static_cast<arrow::FloatBuilder*>(action_b.value_builder());

    for (int64_t i = 0; i < n; ++i) {
        THROW_IF_NOT_OK(state_b.Append());
        for (int d=0; d<14; ++d) THROW_IF_NOT_OK(sf_builder->Append(state[i*14+d]));

        THROW_IF_NOT_OK(action_b.Append());
        for (int d=0; d<14; ++d) THROW_IF_NOT_OK(af_builder->Append(actions[i*14+d]));

        THROW_IF_NOT_OK(ts_b.Append((float)i / cfg.fps));
        THROW_IF_NOT_OK(fi_b.Append(i));
        THROW_IF_NOT_OK(ei_b.Append(ep_idx));
        THROW_IF_NOT_OK(ix_b.Append(ep_start + i));
        THROW_IF_NOT_OK(ti_b.Append(0));

        auto j0 = encode_jpeg(img0.data()+i*h0*w0*3, h0, w0, cfg.jpeg_quality);
        auto j1 = encode_jpeg(img1.data()+i*h1*w1*3, h1, w1, cfg.jpeg_quality);
        auto j2 = encode_jpeg(img2.data()+i*h2*w2*3, h2, w2, cfg.jpeg_quality);
        std::string fn = "frame_" + std::to_string(i) + ".jpg";

        THROW_IF_NOT_OK(img_b.Append());
        THROW_IF_NOT_OK(bytes_b->Append(j0.data(), j0.size()));
        THROW_IF_NOT_OK(path_b->Append(fn));

        THROW_IF_NOT_OK(wrist_b.Append());
        THROW_IF_NOT_OK(wbytes_b->Append(j1.data(), j1.size()));
        THROW_IF_NOT_OK(wpath_b->Append(fn));

        THROW_IF_NOT_OK(wrist_r_b.Append());
        THROW_IF_NOT_OK(rbytes_b->Append(j2.data(), j2.size()));
        THROW_IF_NOT_OK(rpath_b->Append(fn));
    }

    std::shared_ptr<arrow::Array> sa,aa,ta,fia,eia,ixa,tia,ima,wia,wra;
    THROW_IF_NOT_OK(state_b.Finish(&sa));
    THROW_IF_NOT_OK(action_b.Finish(&aa));
    THROW_IF_NOT_OK(ts_b.Finish(&ta));
    THROW_IF_NOT_OK(fi_b.Finish(&fia));
    THROW_IF_NOT_OK(ei_b.Finish(&eia));
    THROW_IF_NOT_OK(ix_b.Finish(&ixa));
    THROW_IF_NOT_OK(ti_b.Finish(&tia));
    THROW_IF_NOT_OK(img_b.Finish(&ima));
    THROW_IF_NOT_OK(wrist_b.Finish(&wia));
    THROW_IF_NOT_OK(wrist_r_b.Finish(&wra));

    auto schema = arrow::schema({
        arrow::field("state", lf_type), arrow::field("actions", lf_type),
        arrow::field("timestamp", arrow::float32()), arrow::field("frame_index", arrow::int64()),
        arrow::field("episode_index", arrow::int64()), arrow::field("index", arrow::int64()),
        arrow::field("task_index", arrow::int64()), arrow::field("image", img_type),
        arrow::field("wrist_image", img_type), arrow::field("wrist_image_right", img_type)});
    schema = schema->WithMetadata(hf_schema_metadata());

    auto table = arrow::Table::Make(schema, {sa,aa,ta,fia,eia,ixa,tia,ima,wia,wra});

    // Write parquet
    fs::path data_dir = fs::path(cfg.output_dir) / "data" / "chunk-000";
    fs::create_directories(data_dir);
    auto fname = "episode_" + std::string(6-std::to_string(ep_idx).size(),'0') + std::to_string(ep_idx) + ".parquet";
    auto out_path = (data_dir / fname).string();

    auto out_file = arrow::io::FileOutputStream::Open(out_path).ValueOrDie();
    auto wp = parquet::WriterProperties::Builder().build();
    auto awp = parquet::ArrowWriterProperties::Builder().store_schema()->build();
    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out_file, 1048576, wp, awp));

    std::cout << "  -> " << out_path << " (" << n << " rows)\n";

    // ---- Stats ----
    EpisodeStats st;
    st.episode_index = ep_idx; st.length = (int)n;
    auto arr_stats = [&](const float* a, int dims, auto& mean, auto& stdv, auto& minv, auto& maxv) {
        mean.resize(dims); stdv.resize(dims); minv.resize(dims); maxv.resize(dims);
        for (int d=0; d<dims; ++d) {
            double s=0,sq=0; float mn=a[d],mx=a[d];
            for (int64_t r=0; r<n; ++r) { float v=a[r*dims+d]; s+=v; sq+=v*v; if(v<mn)mn=v; if(v>mx)mx=v; }
            mean[d]=(float)(s/n); stdv[d]=(float)std::sqrt(std::max(0.0,sq/n-mean[d]*mean[d]));
            minv[d]=mn; maxv[d]=mx;
        }
    };
    arr_stats(state.data(), 14, st.state_mean, st.state_std, st.state_min, st.state_max);
    arr_stats(actions.data(), 14, st.action_mean, st.action_std, st.action_min, st.action_max);
    compute_image_stats(img0.data(), n, h0, w0, st.img0.min, st.img0.max, st.img0.mean, st.img0.std);
    compute_image_stats(img1.data(), n, h1, w1, st.img1.min, st.img1.max, st.img1.mean, st.img1.std);
    compute_image_stats(img2.data(), n, h2, w2, st.img2.min, st.img2.max, st.img2.mean, st.img2.std);
    return st;
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------
void write_metadata(const fs::path& out_dir, const zarr::DatasetArrays& ds,
                    const ConvertConfig& cfg, const std::vector<EpisodeStats>& all_stats)
{
    auto meta = out_dir / "meta";
    fs::create_directories(meta);

    int n_ep = (int)all_stats.size();
    int64_t total_frames = 0;
    for (auto& s : all_stats) total_frames += s.length;

    auto _a14 = [](const std::vector<float>& v) { json j; for (auto x:v) j.push_back(x); return j; };
    auto _img_shape = [](const float v[3]) {
        return json::array({json::array({json::array({v[0]})}),
                            json::array({json::array({v[1]})}),
                            json::array({json::array({v[2]})})}); };

    // info.json
    json info;
    info["codebase_version"] = "v2.1";
    info["robot_type"] = "piper_dual_" + cfg.mode;
    info["total_episodes"] = n_ep; info["total_frames"] = total_frames;
    info["total_tasks"] = 1; info["total_videos"] = 0; info["total_chunks"] = 1;
    info["chunks_size"] = 1000; info["fps"] = cfg.fps;
    info["splits"]["train"] = "0:" + std::to_string(n_ep);
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet";
    info["video_path"] = nullptr;
    info["features"]["state"] = {{"dtype","float32"},{"shape",{14}},{"names",{
        "left_joint_1","left_joint_2","left_joint_3","left_joint_4","left_joint_5","left_joint_6","left_gripper",
        "right_joint_1","right_joint_2","right_joint_3","right_joint_4","right_joint_5","right_joint_6","right_gripper"}}};
    info["features"]["actions"] = {{"dtype","float32"},{"shape",{14}},{"names",{
        "left_joint_1","left_joint_2","left_joint_3","left_joint_4","left_joint_5","left_joint_6","left_gripper",
        "right_joint_1","right_joint_2","right_joint_3","right_joint_4","right_joint_5","right_joint_6","right_gripper"}}};
    info["features"]["timestamp"] = {{"dtype","float32"},{"shape",{1}},{"names",nullptr}};
    info["features"]["frame_index"] = {{"dtype","int64"},{"shape",{1}},{"names",nullptr}};
    info["features"]["episode_index"] = {{"dtype","int64"},{"shape",{1}},{"names",nullptr}};
    info["features"]["index"] = {{"dtype","int64"},{"shape",{1}},{"names",nullptr}};
    info["features"]["task_index"] = {{"dtype","int64"},{"shape",{1}},{"names",nullptr}};
    info["features"]["image"] = {{"dtype","image"},{"shape",{(int)ds.img0.shape[1],(int)ds.img0.shape[2],3}},{"names",{"height","width","channel"}}};
    info["features"]["wrist_image"] = {{"dtype","image"},{"shape",{(int)ds.img1.shape[1],(int)ds.img1.shape[2],3}},{"names",{"height","width","channel"}}};
    info["features"]["wrist_image_right"] = {{"dtype","image"},{"shape",{(int)ds.img2.shape[1],(int)ds.img2.shape[2],3}},{"names",{"height","width","channel"}}};

    std::ofstream(meta / "info.json") << info.dump(2);

    // episodes.jsonl
    std::ofstream epf(meta / "episodes.jsonl");
    for (auto& s : all_stats)
        epf << json{{"episode_index",s.episode_index},{"tasks",{cfg.task_text}},{"length",s.length}}.dump() << "\n";

    // tasks.jsonl
    std::ofstream(meta / "tasks.jsonl") << json{{"task_index",0},{"task",cfg.task_text}}.dump() << "\n";

    // episodes_stats.jsonl
    std::ofstream sf(meta / "episodes_stats.jsonl");
    for (auto& s : all_stats) {
        json st;
        st["state"]["min"]=_a14(s.state_min); st["state"]["max"]=_a14(s.state_max);
        st["state"]["mean"]=_a14(s.state_mean); st["state"]["std"]=_a14(s.state_std);
        st["state"]["count"]={s.length};
        st["actions"]["min"]=_a14(s.action_min); st["actions"]["max"]=_a14(s.action_max);
        st["actions"]["mean"]=_a14(s.action_mean); st["actions"]["std"]=_a14(s.action_std);
        st["actions"]["count"]={s.length};
        st["image"]["min"]=_img_shape(s.img0.min); st["image"]["max"]=_img_shape(s.img0.max);
        st["image"]["mean"]=_img_shape(s.img0.mean); st["image"]["std"]=_img_shape(s.img0.std);
        st["image"]["count"]={s.length};
        st["wrist_image"]["min"]=_img_shape(s.img1.min); st["wrist_image"]["max"]=_img_shape(s.img1.max);
        st["wrist_image"]["mean"]=_img_shape(s.img1.mean); st["wrist_image"]["std"]=_img_shape(s.img1.std);
        st["wrist_image"]["count"]={s.length};
        st["wrist_image_right"]["min"]=_img_shape(s.img2.min); st["wrist_image_right"]["max"]=_img_shape(s.img2.max);
        st["wrist_image_right"]["mean"]=_img_shape(s.img2.mean); st["wrist_image_right"]["std"]=_img_shape(s.img2.std);
        st["wrist_image_right"]["count"]={s.length};
        sf << json{{"episode_index",s.episode_index},{"stats",st}}.dump() << "\n";
    }

    // stats.json
    json gs;
    if (!all_stats.empty()) {
        auto& s0 = all_stats[0];
        gs["state"]["min"]=_a14(s0.state_min); gs["state"]["max"]=_a14(s0.state_max);
        gs["state"]["mean"]=_a14(s0.state_mean); gs["state"]["std"]=_a14(s0.state_std);
        gs["state"]["count"]={total_frames};
        gs["actions"]["min"]=_a14(s0.action_min); gs["actions"]["max"]=_a14(s0.action_max);
        gs["actions"]["mean"]=_a14(s0.action_mean); gs["actions"]["std"]=_a14(s0.action_std);
        gs["actions"]["count"]={total_frames};
    }
    std::ofstream(meta / "stats.json") << gs.dump(2);

    std::cout << "[DONE] Dataset written to " << out_dir << "\n";
}

// ---------------------------------------------------------------------------
// Main converter
// ---------------------------------------------------------------------------
int run_converter(const ConvertConfig& cfg) {
    auto ds = zarr::open_dataset(cfg.zarr_path);
    std::cout << "[INFO] Total frames: " << ds.total_frames << ", Episodes: " << ds.num_episodes << "\n";
    std::cout << "[INFO] Base camera: " << ds.img0.shape[1] << "x" << ds.img0.shape[2] << "\n";

    std::vector<std::tuple<int,int64_t,int64_t>> ranges;
    int64_t ep_start = 0;
    for (int i=0; i<ds.num_episodes; ++i) {
        ranges.emplace_back(i, ep_start, (int64_t)ds.episode_ends[i]);
        ep_start = ds.episode_ends[i];
    }

    int nt = cfg.num_threads > 0 ? cfg.num_threads : (int)std::thread::hardware_concurrency();
    int np = cfg.parallel_episodes > 0 ? cfg.parallel_episodes : std::max(1, nt/2);
    std::cout << "[INFO] Threads: " << nt << ", Parallel episodes: " << np << "\n";

    std::vector<EpisodeStats> all_stats(ds.num_episodes);
    ThreadPool pool(np);
    std::vector<std::future<EpisodeStats>> futures;
    for (auto& [ep, s, e] : ranges)
        futures.push_back(pool.submit([&, ep, s, e]() { return process_episode(ds, cfg, ep, s, e); }));
    for (size_t i=0; i<futures.size(); ++i) all_stats[i] = futures[i].get();

    write_metadata(cfg.output_dir, ds, cfg, all_stats);
    return 0;
}
