#ifndef __SD_CONDITIONING_CONDITIONING_CACHE_H__
#define __SD_CONDITIONING_CONDITIONING_CACHE_H__

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <list>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "conditioning/conditioner.hpp"

struct ConditioningCacheEntryInfo {
    std::string id;
    std::string prompt;
    int width     = -1;
    int height    = -1;
    int clip_skip = -1;
    bool zero_out_masked = false;
    int64_t created_at_unix = 0;
    int64_t age_seconds     = 0;
};

class ConditioningCache {
    struct Entry {
        ConditionerParams params;
        std::vector<sd::Tensor<float>> ref_images;
        sd::PromptInput prompt_input;
        std::vector<MiniMaxH3PresentationItem> references;
        SDCondition condition;
        std::chrono::system_clock::time_point created_at;
        std::string id;
        bool persisted = false;

        Entry(const ConditionerParams& input, const SDCondition& output, std::string entry_id,
              std::chrono::system_clock::time_point created = std::chrono::system_clock::now())
            : params(input), condition(output), created_at(created),
              id(std::move(entry_id)) {
            if (input.prompt_input != nullptr) {
                prompt_input        = *input.prompt_input;
                params.prompt_input = &prompt_input;
            }
            // Request-owned reference pointers must not outlive the request.
            if (input.ref_images != nullptr) {
                ref_images        = *input.ref_images;
                params.ref_images = &ref_images;
            }
            if (input.minimax_h3_references != nullptr) {
                references                   = *input.minimax_h3_references;
                params.minimax_h3_references = &references;
            }
        }

        Entry(const Entry&)            = delete;
        Entry& operator=(const Entry&) = delete;
    };

    size_t capacity_ = 4;
    std::list<Entry> entries_;
    std::filesystem::path storage_dir_;
    std::string storage_scope_;
    bool validate_moss_ = false;
    bool invalid_moss_encoding_ = false;

    bool valid_condition(const SDCondition& condition) const {
        if (!validate_moss_) return true;
        const auto& values = condition.c_crossattn.values();
        return !values.empty() &&
               std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }) &&
               std::any_of(values.begin(), values.end(), [](float v) { return v != 0.f; });
    }

    struct PersistedEntry {
        ConditionerParams params;
        SDCondition condition;
        std::chrono::system_clock::time_point created_at;
        std::string id;
    };

    static bool same_images(const std::vector<sd::Tensor<float>>& a,
                            const std::vector<sd::Tensor<float>>& b) {
        return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                          [](const sd::Tensor<float>& x, const sd::Tensor<float>& y) {
                              return x.shape() == y.shape() && x.values() == y.values();
                          });
    }

    static bool same_params(const ConditionerParams& a, const ConditionerParams& b) {
        const auto fields = [](const ConditionerParams& p) {
            const auto& r = p.ref_image_params;
            return std::tie(p.text, p.clip_skip, p.width, p.height, p.zero_out_masked,
                            r.pass_to_vlm, r.pass_to_dit, r.ref_index_mode,
                            r.force_ref_timestep_zero, r.resize_before_vae, r.vae_input_max_pixels,
                            r.vlm_resize_mode, r.vlm_min_size, r.vlm_max_size, r.resize_vae_to_target,
                            r.resize_single_ref_to_target, r.vae_size_buckets, r.vae_center_crop,
                            r.vlm_use_vae_input, r.ref_image_size);
        };
        if (fields(a) != fields(b) ||
            (a.prompt_input == nullptr) != (b.prompt_input == nullptr) ||
            (a.ref_images == nullptr) != (b.ref_images == nullptr) ||
            (a.minimax_h3_references == nullptr) != (b.minimax_h3_references == nullptr)) {
            return false;
        }
        if (a.prompt_input != nullptr &&
            !std::equal(a.prompt_input->begin(), a.prompt_input->end(),
                        b.prompt_input->begin(), b.prompt_input->end(),
                        [](const sd::PromptPart& x, const sd::PromptPart& y) {
                            return x.type == y.type && x.content == y.content && x.weight == y.weight &&
                                   (x.image == y.image ||
                                    (x.image && y.image && x.image->shape() == y.image->shape() && x.image->values() == y.image->values()));
                        })) {
            return false;
        }
        if (a.ref_images != nullptr && !same_images(*a.ref_images, *b.ref_images)) {
            return false;
        }
        if (a.minimax_h3_references != nullptr &&
            !std::equal(a.minimax_h3_references->begin(), a.minimax_h3_references->end(),
                        b.minimax_h3_references->begin(), b.minimax_h3_references->end(),
                        [](const MiniMaxH3PresentationItem& x, const MiniMaxH3PresentationItem& y) {
                            return x.kind == y.kind && x.timestamps == y.timestamps && same_images(x.frames, y.frames);
                        })) {
            return false;
        }
        return true;
    }

    static std::string make_id(const ConditionerParams& params);

    static int64_t unix_seconds(std::chrono::system_clock::time_point tp) {
        return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
    }

    std::filesystem::path file_for_id(const std::string& id) const;
    std::filesystem::path role_file_for_id(const std::string& id) const;
    bool read_entry(const std::filesystem::path& path, PersistedEntry& out) const;
    bool load_for_params(const ConditionerParams& params);
    void load_existing();
    void persist(Entry& entry);
    void remove_persisted(const Entry& entry);
    void touch_persisted(const Entry& entry);

public:
    void set_moss_validation(bool enabled) {
        validate_moss_ = enabled;
        if (enabled) {
            for (auto it = entries_.begin(); it != entries_.end();) {
                if (!valid_condition(it->condition)) {
                    LOG_WARN("conditioning cache: ignoring invalid MOSS text encoding '%s'", it->id.c_str());
                    it = entries_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    void set_capacity(size_t capacity) {
        capacity_ = capacity;
        while (entries_.size() > capacity_) {
            remove_persisted(entries_.back());
            entries_.pop_back();
        }
    }

    void configure_persistence(const std::string& directory, const std::string& scope);
    size_t capacity() const { return capacity_; }
    size_t size() const { return entries_.size(); }
    std::string storage_scope_id() const { return storage_dir_.filename().string(); }
    bool invalid_moss_encoding() const { return invalid_moss_encoding_; }

    void clear() {
        for (const auto& entry : entries_) {
            remove_persisted(entry);
        }
        entries_.clear();
        std::error_code ec;
        if (!storage_dir_.empty()) {
            for (const auto& item : std::filesystem::directory_iterator(storage_dir_, ec)) {
                if (ec) {
                    break;
                }
                if (item.path().extension() == ".cndcache" || item.path().extension() == ".tmp" || item.path().extension() == ".roles") {
                    std::filesystem::remove(item.path(), ec);
                    ec.clear();
                }
            }
        }
    }

    void clear_memory() {
        entries_.clear();
    }

    bool erase_id(const std::string& id) {
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->id == id) {
                remove_persisted(*it);
                entries_.erase(it);
                return true;
            }
        }
        return false;
    }

    size_t purge_older_than(std::chrono::seconds max_age);
    void record_role(const ConditionerParams& params, int role);

    std::vector<ConditioningCacheEntryInfo> list() const {
        std::vector<ConditioningCacheEntryInfo> out;
        out.reserve(entries_.size());
        const auto now = std::chrono::system_clock::now();
        for (const auto& entry : entries_) {
            ConditioningCacheEntryInfo info;
            info.id               = entry.id;
            info.prompt           = entry.params.text;
            info.width            = entry.params.width;
            info.height           = entry.params.height;
            info.clip_skip        = entry.params.clip_skip;
            info.zero_out_masked  = entry.params.zero_out_masked;
            info.created_at_unix  = unix_seconds(entry.created_at);
            info.age_seconds      = std::chrono::duration_cast<std::chrono::seconds>(now - entry.created_at).count();
            out.push_back(std::move(info));
        }
        return out;
    }

    bool contains(const ConditionerParams& params) {
        if (capacity_ == 0) {
            return false;
        }
        for (const auto& entry : entries_) {
            if (same_params(entry.params, params)) {
                return true;
            }
        }
        return load_for_params(params);
    }

    SDCondition get(Conditioner& conditioner, int n_threads, const ConditionerParams& params) {
        invalid_moss_encoding_ = false;
        if (capacity_ != 0) {
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                if (same_params(it->params, params)) {
                    entries_.splice(entries_.begin(), entries_, it);
                    touch_persisted(entries_.front());
                    LOG_INFO("conditioning cache hit");
                    return entries_.front().condition;
                }
            }
            if (load_for_params(params)) {
                LOG_INFO("conditioning cache hit (disk)");
                return entries_.front().condition;
            }
        }
        auto condition = conditioner.get_learned_condition(n_threads, params);
        if (validate_moss_ && !valid_condition(condition)) {
            invalid_moss_encoding_ = true;
            LOG_ERROR("INVALID_MOSS_TEXT_ENCODING: text encoder returned empty, all-zero, or non-finite embeddings");
            return {};
        }
        if (capacity_ != 0 && !condition.empty()) {
            if (entries_.size() == capacity_) {
                remove_persisted(entries_.back());
                entries_.pop_back();
            }
            entries_.emplace_front(params, condition, make_id(params));
            persist(entries_.front());
            LOG_VERBOSE("conditioning cache stored (%zu/%zu)", entries_.size(), capacity_);
        }
        return condition;
    }
};

#endif  // __SD_CONDITIONING_CONDITIONING_CACHE_H__
