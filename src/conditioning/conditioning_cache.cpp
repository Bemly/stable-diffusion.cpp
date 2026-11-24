#include "conditioning_cache.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <system_error>

namespace {

constexpr std::array<char, 8> kMagic{'S', 'D', 'C', 'A', 'C', 'H', 'E', '1'};
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kEndianMarker = 0x01020304;
constexpr uint64_t kMaxFileBytes = 512ULL * 1024 * 1024;
constexpr uint64_t kMaxElements = kMaxFileBytes / sizeof(float);
constexpr uint64_t kMaxStringBytes = 16ULL * 1024 * 1024;
constexpr uint64_t kMaxVectorItems = 1ULL * 1024 * 1024;

template <typename T>
bool write_value(std::ostream& out, const T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "cache values must be trivially copyable");
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(out);
}

template <typename T>
bool read_value(std::istream& in, T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "cache values must be trivially copyable");
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(in);
}

bool write_string(std::ostream& out, const std::string& value) {
    if (value.size() > kMaxStringBytes) {
        return false;
    }
    const uint64_t size = value.size();
    return write_value(out, size) &&
           (size == 0 || (out.write(value.data(), static_cast<std::streamsize>(size)), static_cast<bool>(out)));
}

bool read_string(std::istream& in, std::string& value) {
    uint64_t size = 0;
    if (!read_value(in, size) || size > kMaxStringBytes) {
        return false;
    }
    value.resize(static_cast<size_t>(size));
    if (size != 0) {
        in.read(value.data(), static_cast<std::streamsize>(size));
    }
    return static_cast<bool>(in);
}

bool write_bool(std::ostream& out, bool value) {
    const uint8_t encoded = value ? 1 : 0;
    return write_value(out, encoded);
}

bool read_bool(std::istream& in, bool& value) {
    uint8_t encoded = 0;
    if (!read_value(in, encoded) || encoded > 1) {
        return false;
    }
    value = encoded != 0;
    return true;
}

template <typename T>
bool write_tensor(std::ostream& out, const sd::Tensor<T>& tensor) {
    if (tensor.shape().size() > 16 || tensor.values().size() > kMaxElements) {
        return false;
    }
    const uint32_t dimensions = static_cast<uint32_t>(tensor.shape().size());
    const uint64_t elements   = tensor.values().size();
    if (!write_value(out, dimensions)) {
        return false;
    }
    for (int64_t dim : tensor.shape()) {
        if (dim < 0 || !write_value(out, dim)) {
            return false;
        }
    }
    if (!write_value(out, elements)) {
        return false;
    }
    if (elements != 0) {
        const auto bytes = static_cast<std::streamsize>(elements * sizeof(T));
        out.write(reinterpret_cast<const char*>(tensor.data()), bytes);
    }
    return static_cast<bool>(out);
}

template <typename T>
bool read_tensor(std::istream& in, sd::Tensor<T>& tensor) {
    uint32_t dimensions = 0;
    if (!read_value(in, dimensions) || dimensions > 16) {
        return false;
    }
    std::vector<int64_t> shape(dimensions);
    uint64_t expected = dimensions == 0 ? 0 : 1;
    for (auto& dim : shape) {
        if (!read_value(in, dim) || dim < 0 ||
            (dim != 0 && expected > kMaxElements / static_cast<uint64_t>(dim))) {
            return false;
        }
        expected *= static_cast<uint64_t>(dim);
    }
    uint64_t elements = 0;
    if (!read_value(in, elements) || elements != expected || elements > kMaxElements) {
        return false;
    }
    try {
        tensor.resize(std::move(shape));
    } catch (const std::exception&) {
        return false;
    }
    if (elements != 0) {
        const auto bytes = static_cast<std::streamsize>(elements * sizeof(T));
        in.read(reinterpret_cast<char*>(tensor.data()), bytes);
    }
    return static_cast<bool>(in);
}

template <typename T>
bool write_tensors(std::ostream& out, const std::vector<sd::Tensor<T>>& tensors) {
    if (tensors.size() > kMaxVectorItems) {
        return false;
    }
    const uint64_t size = tensors.size();
    if (!write_value(out, size)) {
        return false;
    }
    for (const auto& tensor : tensors) {
        if (!write_tensor(out, tensor)) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool read_tensors(std::istream& in, std::vector<sd::Tensor<T>>& tensors) {
    uint64_t size = 0;
    if (!read_value(in, size) || size > kMaxVectorItems) {
        return false;
    }
    tensors.resize(static_cast<size_t>(size));
    for (auto& tensor : tensors) {
        if (!read_tensor(in, tensor)) {
            return false;
        }
    }
    return true;
}

bool write_params(std::ostream& out, const ConditionerParams& params) {
    const auto& r = params.ref_image_params;
    const int32_t ref_index_mode = static_cast<int32_t>(r.ref_index_mode);
    const int32_t vlm_resize_mode = static_cast<int32_t>(r.vlm_resize_mode);
    const int32_t ref_image_size = static_cast<int32_t>(r.ref_image_size);
    return write_string(out, params.text) &&
           write_value(out, params.clip_skip) &&
           write_value(out, params.width) &&
           write_value(out, params.height) &&
           write_bool(out, params.zero_out_masked) &&
           write_bool(out, r.pass_to_vlm) &&
           write_bool(out, r.pass_to_dit) &&
           write_value(out, ref_index_mode) &&
           write_bool(out, r.force_ref_timestep_zero) &&
           write_bool(out, r.resize_before_vae) &&
           write_value(out, r.vae_input_max_pixels) &&
           write_value(out, vlm_resize_mode) &&
           write_value(out, r.vlm_min_size) &&
           write_value(out, r.vlm_max_size) &&
           write_bool(out, r.resize_vae_to_target) &&
           write_bool(out, r.resize_single_ref_to_target) &&
           write_bool(out, r.vae_size_buckets) &&
           write_bool(out, r.vae_center_crop) &&
           write_bool(out, r.vlm_use_vae_input) &&
           write_value(out, ref_image_size);
}

bool read_params(std::istream& in, ConditionerParams& params) {
    auto& r = params.ref_image_params;
    int32_t ref_index_mode = 0;
    int32_t vlm_resize_mode = 0;
    int32_t ref_image_size = 0;
    if (!read_string(in, params.text) ||
        !read_value(in, params.clip_skip) ||
        !read_value(in, params.width) ||
        !read_value(in, params.height) ||
        !read_bool(in, params.zero_out_masked) ||
        !read_bool(in, r.pass_to_vlm) ||
        !read_bool(in, r.pass_to_dit) ||
        !read_value(in, ref_index_mode) ||
        !read_bool(in, r.force_ref_timestep_zero) ||
        !read_bool(in, r.resize_before_vae) ||
        !read_value(in, r.vae_input_max_pixels) ||
        !read_value(in, vlm_resize_mode) ||
        !read_value(in, r.vlm_min_size) ||
        !read_value(in, r.vlm_max_size) ||
        !read_bool(in, r.resize_vae_to_target) ||
        !read_bool(in, r.resize_single_ref_to_target) ||
        !read_bool(in, r.vae_size_buckets) ||
        !read_bool(in, r.vae_center_crop) ||
        !read_bool(in, r.vlm_use_vae_input) ||
        !read_value(in, ref_image_size)) {
        return false;
    }
    r.ref_index_mode = static_cast<Rope::RefIndexMode>(ref_index_mode);
    r.vlm_resize_mode = static_cast<RefImageResizeMode>(vlm_resize_mode);
    r.ref_image_size = static_cast<RefImageSizeMode>(ref_image_size);
    return true;
}

bool write_condition(std::ostream& out, const SDCondition& c) {
    if (!write_tensor(out, c.c_crossattn) ||
        !write_tensor(out, c.c_vector) ||
        !write_tensor(out, c.c_concat) ||
        !write_tensor(out, c.c_t5_ids) ||
        !write_tensor(out, c.c_t5_weights) ||
        !write_tensor(out, c.c_input_ids) ||
        !write_tensor(out, c.c_position_ids) ||
        !write_tensor(out, c.c_token_types) ||
        !write_tensor(out, c.c_vinput_mask) ||
        c.c_image_embeds.size() > kMaxVectorItems) {
        return false;
    }
    const uint64_t image_embed_count = c.c_image_embeds.size();
    if (!write_value(out, image_embed_count)) {
        return false;
    }
    for (const auto& item : c.c_image_embeds) {
        if (!write_value(out, item.first) || !write_tensor(out, item.second)) {
            return false;
        }
    }
    if (!write_tensors(out, c.c_ref_images) || !write_tensors(out, c.c_ref_audios) ||
        c.c_reference_blocks.size() > kMaxVectorItems) {
        return false;
    }
    const uint64_t reference_count = c.c_reference_blocks.size();
    if (!write_value(out, reference_count)) {
        return false;
    }
    for (const auto& block : c.c_reference_blocks) {
        const int32_t kind = static_cast<int32_t>(block.kind);
        if (!write_value(out, kind) || !write_value(out, block.video_index) ||
            !write_value(out, block.audio_index)) {
            return false;
        }
    }
    return write_tensors(out, c.extra_c_crossattns);
}

bool read_condition(std::istream& in, SDCondition& c) {
    if (!read_tensor(in, c.c_crossattn) ||
        !read_tensor(in, c.c_vector) ||
        !read_tensor(in, c.c_concat) ||
        !read_tensor(in, c.c_t5_ids) ||
        !read_tensor(in, c.c_t5_weights) ||
        !read_tensor(in, c.c_input_ids) ||
        !read_tensor(in, c.c_position_ids) ||
        !read_tensor(in, c.c_token_types) ||
        !read_tensor(in, c.c_vinput_mask)) {
        return false;
    }
    uint64_t image_embed_count = 0;
    if (!read_value(in, image_embed_count) || image_embed_count > kMaxVectorItems) {
        return false;
    }
    c.c_image_embeds.resize(static_cast<size_t>(image_embed_count));
    for (auto& item : c.c_image_embeds) {
        if (!read_value(in, item.first) || !read_tensor(in, item.second)) {
            return false;
        }
    }
    if (!read_tensors(in, c.c_ref_images) || !read_tensors(in, c.c_ref_audios)) {
        return false;
    }
    uint64_t reference_count = 0;
    if (!read_value(in, reference_count) || reference_count > kMaxVectorItems) {
        return false;
    }
    c.c_reference_blocks.resize(static_cast<size_t>(reference_count));
    for (auto& block : c.c_reference_blocks) {
        int32_t kind = 0;
        if (!read_value(in, kind) || !read_value(in, block.video_index) ||
            !read_value(in, block.audio_index)) {
            return false;
        }
        block.kind = static_cast<MiniMaxH3ReferenceKind>(kind);
    }
    return read_tensors(in, c.extra_c_crossattns);
}

bool read_created_at(const std::filesystem::path& path, int64_t& created_at) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::array<char, kMagic.size()> magic{};
    uint32_t version = 0;
    uint32_t endian = 0;
    return static_cast<bool>(
        in.read(magic.data(), static_cast<std::streamsize>(magic.size())) &&
        magic == kMagic &&
        read_value(in, version) && version == kFormatVersion &&
        read_value(in, endian) && endian == kEndianMarker &&
        read_value(in, created_at));
}

bool persistable(const ConditionerParams& params, const SDCondition& condition) {
    return params.prompt_input == nullptr && params.ref_images == nullptr && params.minimax_h3_references == nullptr &&
           condition.c_image_embeds.empty() && condition.c_ref_images.empty() &&
           condition.c_ref_audios.empty() && condition.c_reference_blocks.empty();
}

uint64_t stable_hash(const std::string& value) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hash_hex(const std::string& value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << stable_hash(value);
    return out.str();
}

void hash_bytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
}

template <typename T>
void hash_value(uint64_t& hash, const T& value) {
    hash_bytes(hash, &value, sizeof(value));
}

void hash_tensor(uint64_t& hash, const sd::Tensor<float>& tensor) {
    for (int64_t dim : tensor.shape()) {
        hash_value(hash, dim);
    }
    const uint64_t size = tensor.values().size();
    hash_value(hash, size);
    if (size != 0) {
        hash_bytes(hash, tensor.data(), static_cast<size_t>(size) * sizeof(float));
    }
}

}  // namespace

std::string ConditioningCache::make_id(const ConditionerParams& params) {
    uint64_t hash = 14695981039346656037ULL;
    const uint64_t text_size = params.text.size();
    hash_value(hash, text_size);
    hash_bytes(hash, params.text.data(), params.text.size());
    hash_value(hash, params.clip_skip);
    hash_value(hash, params.width);
    hash_value(hash, params.height);
    hash_value(hash, params.zero_out_masked);
    const auto& r = params.ref_image_params;
    hash_value(hash, r.pass_to_vlm);
    hash_value(hash, r.pass_to_dit);
    hash_value(hash, r.ref_index_mode);
    hash_value(hash, r.force_ref_timestep_zero);
    hash_value(hash, r.resize_before_vae);
    hash_value(hash, r.vae_input_max_pixels);
    hash_value(hash, r.vlm_resize_mode);
    hash_value(hash, r.vlm_min_size);
    hash_value(hash, r.vlm_max_size);
    hash_value(hash, r.resize_vae_to_target);
    hash_value(hash, r.resize_single_ref_to_target);
    hash_value(hash, r.vae_size_buckets);
    hash_value(hash, r.vae_center_crop);
    hash_value(hash, r.vlm_use_vae_input);
    hash_value(hash, r.ref_image_size);
    if (params.ref_images != nullptr) {
        for (const auto& image : *params.ref_images) {
            hash_tensor(hash, image);
        }
    }
    if (params.minimax_h3_references != nullptr) {
        for (const auto& reference : *params.minimax_h3_references) {
            hash_value(hash, reference.kind);
            for (float timestamp : reference.timestamps) {
                hash_value(hash, timestamp);
            }
            for (const auto& frame : reference.frames) {
                hash_tensor(hash, frame);
            }
        }
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

std::filesystem::path ConditioningCache::file_for_id(const std::string& id) const {
    return storage_dir_ / (id + ".cndcache");
}

std::filesystem::path ConditioningCache::role_file_for_id(const std::string& id) const {
    return storage_dir_ / (id + ".roles");
}

void ConditioningCache::record_role(const ConditionerParams& params, int role) {
    if (role != 1 && role != 2) return;
    auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& entry) {
        return same_params(entry.params, params);
    });
    if (it == entries_.end() || !it->persisted || storage_dir_.empty()) return;
    const auto path = role_file_for_id(it->id);
    int roles = 0;
    {
        std::ifstream input(path);
        if (input) input >> roles;
    }
    if ((roles & role) != 0) return;
    roles |= role;
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        output << roles;
        output.close();
        if (!output) {
            LOG_WARN("conditioning cache: failed to write prompt roles for '%s'", it->id.c_str());
            return;
        }
    }
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    if (ec) LOG_WARN("conditioning cache: failed to save prompt roles for '%s': %s", it->id.c_str(), ec.message().c_str());
}

void ConditioningCache::configure_persistence(const std::string& directory, const std::string& scope) {
    clear_memory();
    storage_dir_.clear();
    storage_scope_.clear();
    if (directory.empty() || scope.empty()) {
        return;
    }
    const std::filesystem::path target = std::filesystem::path(directory) / hash_hex(scope);
    std::error_code ec;
    std::filesystem::create_directories(target, ec);
    if (ec) {
        LOG_WARN("conditioning cache: cannot create persistence directory '%s': %s",
                 target.string().c_str(), ec.message().c_str());
        return;
    }
    storage_dir_ = target;
    storage_scope_ = scope;
    load_existing();
    if (!entries_.empty()) {
        LOG_INFO("conditioning cache restored from disk (%zu/%zu)", entries_.size(), capacity_);
    }
}

bool ConditioningCache::read_entry(const std::filesystem::path& path, PersistedEntry& out) const {
    try {
        std::error_code ec;
        const uint64_t size = std::filesystem::file_size(path, ec);
        if (ec || size > kMaxFileBytes) {
            return false;
        }
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return false;
        }
        std::array<char, kMagic.size()> magic{};
        uint32_t version = 0;
        uint32_t endian = 0;
        int64_t created_at = 0;
        std::string scope;
        if (!in.read(magic.data(), static_cast<std::streamsize>(magic.size())) ||
            magic != kMagic ||
            !read_value(in, version) || version != kFormatVersion ||
            !read_value(in, endian) || endian != kEndianMarker ||
            !read_value(in, created_at) ||
            !read_string(in, scope) || scope != storage_scope_ ||
            !read_string(in, out.id) || !read_params(in, out.params) ||
            !read_condition(in, out.condition) || in.peek() != std::char_traits<char>::eof()) {
            return false;
        }
        out.created_at = std::chrono::system_clock::time_point(std::chrono::seconds(created_at));
        return !out.condition.empty() && persistable(out.params, out.condition) &&
               out.id == make_id(out.params);
    } catch (const std::exception&) {
        return false;
    }
}

void ConditioningCache::load_existing() {
    if (capacity_ == 0 || storage_dir_.empty()) {
        return;
    }
    struct Candidate {
        std::filesystem::path path;
        std::filesystem::file_time_type modified;
    };
    std::vector<Candidate> files;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(storage_dir_, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().extension() != ".cndcache") {
            continue;
        }
        const auto modified = it->last_write_time(ec);
        if (!ec) {
            files.push_back({it->path(), modified});
        } else {
            ec.clear();
        }
    }
    std::sort(files.begin(), files.end(), [](const Candidate& a, const Candidate& b) {
        return a.modified > b.modified;
    });
    for (size_t i = 0; i < files.size(); ++i) {
        const auto& file = files[i];
        if (entries_.size() == capacity_) {
            for (; i < files.size(); ++i) {
                std::filesystem::remove(files[i].path, ec);
                ec.clear();
                std::filesystem::remove(files[i].path.parent_path() / (files[i].path.stem().string() + ".roles"), ec);
                ec.clear();
            }
            break;
        }
        PersistedEntry saved;
        if (!read_entry(file.path, saved)) {
            LOG_WARN("conditioning cache: ignoring invalid persisted entry '%s'", file.path.string().c_str());
            std::filesystem::remove(file.path, ec);
            ec.clear();
            std::filesystem::remove(file.path.parent_path() / (file.path.stem().string() + ".roles"), ec);
            ec.clear();
            continue;
        }
        if (!valid_condition(saved.condition)) {
            LOG_WARN("conditioning cache: ignoring invalid MOSS text encoding '%s'", saved.id.c_str());
            continue;
        }
        const bool duplicate = std::any_of(entries_.begin(), entries_.end(), [&](const Entry& entry) {
            return entry.id == saved.id;
        });
        if (duplicate) {
            continue;
        }
        entries_.emplace_back(saved.params, saved.condition, saved.id, saved.created_at);
        entries_.back().persisted = true;
    }
}

bool ConditioningCache::load_for_params(const ConditionerParams& params) {
    if (capacity_ == 0 || storage_dir_.empty()) {
        return false;
    }
    const std::string id = make_id(params);
    PersistedEntry saved;
    const auto path = file_for_id(id);
    if (!read_entry(path, saved) || !same_params(saved.params, params)) {
        return false;
    }
    if (!valid_condition(saved.condition)) {
        LOG_WARN("conditioning cache: ignoring invalid MOSS text encoding '%s'", saved.id.c_str());
        return false;
    }
    if (entries_.size() == capacity_) {
        remove_persisted(entries_.back());
        entries_.pop_back();
    }
    entries_.emplace_front(saved.params, saved.condition, saved.id, saved.created_at);
    entries_.front().persisted = true;
    touch_persisted(entries_.front());
    return true;
}

void ConditioningCache::persist(Entry& entry) {
    if (storage_dir_.empty() || !persistable(entry.params, entry.condition)) {
        return;
    }
    std::error_code ec;
    const auto path = file_for_id(entry.id);
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        const uint32_t version = kFormatVersion;
        const int64_t created_at = unix_seconds(entry.created_at);
        const bool ok = out &&
                        out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size())) &&
                        write_value(out, version) &&
                        write_value(out, kEndianMarker) &&
                        write_value(out, created_at) &&
                        write_string(out, storage_scope_) &&
                        write_string(out, entry.id) &&
                        write_params(out, entry.params) &&
                        write_condition(out, entry.condition);
        if (!ok) {
            out.close();
            std::filesystem::remove(temporary, ec);
            LOG_WARN("conditioning cache: failed to serialize entry '%s'", entry.id.c_str());
            return;
        }
        out.flush();
        if (!out) {
            out.close();
            std::filesystem::remove(temporary, ec);
            LOG_WARN("conditioning cache: failed to write entry '%s'", entry.id.c_str());
            return;
        }
    }
    const uint64_t file_size = std::filesystem::file_size(temporary, ec);
    if (ec || file_size > kMaxFileBytes) {
        std::filesystem::remove(temporary, ec);
        LOG_WARN("conditioning cache: entry '%s' exceeds the persistence size limit", entry.id.c_str());
        return;
    }
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        const std::string first_error = ec.message();
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        ec.clear();
        std::filesystem::rename(temporary, path, ec);
        if (ec && first_error.size()) {
            LOG_DEBUG("conditioning cache: replacing existing entry '%s' after rename error: %s",
                      entry.id.c_str(), first_error.c_str());
        }
    }
    if (ec) {
        const std::string message = ec.message();
        std::filesystem::remove(temporary, ec);
        LOG_WARN("conditioning cache: failed to commit entry '%s': %s",
                 entry.id.c_str(), message.c_str());
        return;
    }
    entry.persisted = true;
}

void ConditioningCache::remove_persisted(const Entry& entry) {
    if (!entry.persisted || storage_dir_.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::remove(file_for_id(entry.id), ec);
    if (ec) {
        LOG_WARN("conditioning cache: failed to remove entry '%s': %s",
                 entry.id.c_str(), ec.message().c_str());
    }
    ec.clear();
    std::filesystem::remove(role_file_for_id(entry.id), ec);
    if (ec) LOG_WARN("conditioning cache: failed to remove prompt roles for '%s': %s", entry.id.c_str(), ec.message().c_str());
}

void ConditioningCache::touch_persisted(const Entry& entry) {
    if (!entry.persisted || storage_dir_.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::last_write_time(file_for_id(entry.id),
                                     std::filesystem::file_time_type::clock::now(), ec);
    if (ec) {
        LOG_WARN("conditioning cache: failed to update LRU time for entry '%s': %s",
                 entry.id.c_str(), ec.message().c_str());
    }
}

size_t ConditioningCache::purge_older_than(std::chrono::seconds max_age) {
    if (max_age.count() < 0) {
        return 0;
    }
    const auto cutoff = std::chrono::system_clock::now() - max_age;
    size_t removed = 0;
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->created_at < cutoff) {
            remove_persisted(*it);
            it = entries_.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }
    std::error_code ec;
    for (std::filesystem::directory_iterator it(storage_dir_, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (it->path().extension() != ".cndcache") {
            continue;
        }
        int64_t created_at = 0;
        if (!read_created_at(it->path(), created_at)) {
            continue;
        }
        const auto created = std::chrono::system_clock::time_point(std::chrono::seconds(created_at));
        if (created < cutoff) {
            std::filesystem::remove(it->path(), ec);
            ec.clear();
            std::filesystem::remove(it->path().parent_path() / (it->path().stem().string() + ".roles"), ec);
            ec.clear();
            ++removed;
        }
    }
    return removed;
}
