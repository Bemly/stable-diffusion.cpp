#ifndef __SD_CONDITIONING_CONDITIONING_CACHE_H__
#define __SD_CONDITIONING_CONDITIONING_CACHE_H__

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <list>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "conditioning/conditioner.hpp"

struct ConditioningCacheEntryInfo {
    std::string id;
    std::string prompt;
    int width     = -1;
    int height    = -1;
    int clip_skip = -1;
    int64_t created_at_unix = 0;
    int64_t age_seconds     = 0;
};

class ConditioningCache {
    struct Entry {
        ConditionerParams params;
        std::vector<sd::Tensor<float>> ref_images;
        std::vector<MiniMaxH3PresentationItem> references;
        SDCondition condition;
        std::chrono::system_clock::time_point created_at;
        std::string id;

        Entry(const ConditionerParams& input, const SDCondition& output, std::string entry_id)
            : params(input), condition(output),
              created_at(std::chrono::system_clock::now()),
              id(std::move(entry_id)) {
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
                            r.ref_image_size);
        };
        if (fields(a) != fields(b) ||
            (a.ref_images == nullptr) != (b.ref_images == nullptr) ||
            (a.minimax_h3_references == nullptr) != (b.minimax_h3_references == nullptr)) {
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

    static std::string make_id(const ConditionerParams& params) {
        // Stable enough for UI delete; collisions across ref-image variants are rare here.
        std::ostringstream oss;
        oss << std::hex;
        size_t h = std::hash<std::string>{}(params.text);
        h ^= (static_cast<size_t>(params.width) << 1);
        h ^= (static_cast<size_t>(params.height) << 3);
        h ^= (static_cast<size_t>(params.clip_skip) << 5);
        h ^= params.zero_out_masked ? 0x9e3779b97f4a7c15ULL : 0;
        oss << h;
        return oss.str();
    }

    static int64_t unix_seconds(std::chrono::system_clock::time_point tp) {
        return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
    }

public:
    void set_capacity(size_t capacity) {
        capacity_ = capacity;
        while (entries_.size() > capacity_) {
            entries_.pop_back();
        }
    }

    size_t capacity() const { return capacity_; }
    size_t size() const { return entries_.size(); }

    void clear() {
        entries_.clear();
    }

    bool erase_id(const std::string& id) {
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->id == id) {
                entries_.erase(it);
                return true;
            }
        }
        return false;
    }

    size_t purge_older_than(std::chrono::seconds max_age) {
        if (max_age.count() < 0) {
            return 0;
        }
        const auto cutoff = std::chrono::system_clock::now() - max_age;
        size_t removed    = 0;
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->created_at < cutoff) {
                it = entries_.erase(it);
                ++removed;
            } else {
                ++it;
            }
        }
        return removed;
    }

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
            info.created_at_unix  = unix_seconds(entry.created_at);
            info.age_seconds      = std::chrono::duration_cast<std::chrono::seconds>(now - entry.created_at).count();
            out.push_back(std::move(info));
        }
        return out;
    }

    bool contains(const ConditionerParams& params) const {
        if (capacity_ == 0) {
            return false;
        }
        for (const auto& entry : entries_) {
            if (same_params(entry.params, params)) {
                return true;
            }
        }
        return false;
    }

    SDCondition get(Conditioner& conditioner, int n_threads, const ConditionerParams& params) {
        if (capacity_ == 0) {
            return conditioner.get_learned_condition(n_threads, params);
        }
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (same_params(it->params, params)) {
                entries_.splice(entries_.begin(), entries_, it);
                LOG_INFO("conditioning cache hit");
                return entries_.front().condition;
            }
        }
        auto condition = conditioner.get_learned_condition(n_threads, params);
        if (!condition.empty()) {
            if (entries_.size() == capacity_) {
                entries_.pop_back();
            }
            entries_.emplace_front(params, condition, make_id(params));
            LOG_VERBOSE("conditioning cache stored (%zu/%zu)", entries_.size(), capacity_);
        }
        return condition;
    }
};

#endif  // __SD_CONDITIONING_CONDITIONING_CACHE_H__
