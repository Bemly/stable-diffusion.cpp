#include "conditioning/conditioning_cache.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class FakeConditioner final : public Conditioner {
public:
    int calls = 0;

    SDCondition get_learned_condition(int, const ConditionerParams&) override {
        ++calls;
        SDCondition condition;
        condition.c_crossattn = sd::Tensor<float>({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
        condition.c_input_ids = sd::Tensor<int32_t>({3}, {11, 12, 13});
        return condition;
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>&) override {}
    void set_flash_attention_enabled(bool) override {}
};

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("sd-conditioning-cache-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    ~TemporaryDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

}  // namespace

int main() {
    try {
        TemporaryDirectory temp;
        ConditionerParams params;
        params.text = "persisted prompt";
        params.width = 768;
        params.height = 1024;
        FakeConditioner conditioner;

        {
            ConditioningCache cache;
            cache.set_capacity(2);
            cache.configure_persistence(temp.path.string(), "model scope");
            const auto result = cache.get(conditioner, 1, params);
            require(conditioner.calls == 1, "cache miss should encode once");
            require(result.c_crossattn.values() == std::vector<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}),
                    "fresh conditioning mismatch");
        }

        std::filesystem::path cache_dir;
        for (const auto& item : std::filesystem::directory_iterator(temp.path)) {
            if (item.is_directory()) {
                cache_dir = item.path();
                break;
            }
        }
        require(!cache_dir.empty(), "persistence directory should be created");
        const auto corrupt_path = cache_dir / "broken.cndcache";
        {
            std::ofstream corrupt(corrupt_path, std::ios::binary);
            corrupt << "broken";
        }

        {
            ConditioningCache restored;
            restored.set_capacity(2);
            restored.configure_persistence(temp.path.string(), "model scope");
            require(restored.size() == 1, "entry should load from disk after restart");
            require(!std::filesystem::exists(corrupt_path), "corrupt entry should be removed");
            require(restored.contains(params), "restored params should hit");
            const auto result = restored.get(conditioner, 1, params);
            require(conditioner.calls == 1, "restored hit must not rerun the encoder");
            require(result.c_crossattn.values() == std::vector<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}),
                    "restored conditioning mismatch");

            restored.clear();
            require(restored.size() == 0, "clear should empty memory");
        }

        {
            ConditioningCache cleared;
            cleared.set_capacity(2);
            cleared.configure_persistence(temp.path.string(), "model scope");
            require(cleared.size() == 0, "clear should delete persisted entries");
        }

        {
            ConditioningCache isolated;
            isolated.set_capacity(2);
            isolated.configure_persistence(temp.path.string(), "different model scope");
            require(isolated.size() == 0, "different model scope must not reuse entries");
        }

        {
            ConditioningCache cache;
            cache.set_capacity(2);
            cache.configure_persistence(temp.path.string(), "lru model scope");
            for (const char* text : {"old prompt", "middle prompt", "new prompt"}) {
                auto variant = params;
                variant.text = text;
                cache.get(conditioner, 1, variant);
            }
            require(cache.size() == 2, "capacity should evict the oldest entry");

            ConditioningCache restored;
            restored.set_capacity(2);
            restored.configure_persistence(temp.path.string(), "lru model scope");
            require(restored.size() == 2, "only the newest entries should survive restart");
            auto oldest = params;
            oldest.text = "old prompt";
            auto middle = params;
            middle.text = "middle prompt";
            auto newest = params;
            newest.text = "new prompt";
            require(!restored.contains(oldest), "evicted entry should not be restored");
            require(restored.contains(middle) && restored.contains(newest),
                    "recent entries should be restored");
            require(restored.purge_older_than(std::chrono::seconds::zero()) == 2,
                    "purge should remove persisted entries");

            ConditioningCache purged;
            purged.set_capacity(2);
            purged.configure_persistence(temp.path.string(), "lru model scope");
            require(purged.size() == 0, "purge should delete entries from disk");
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "conditioning-cache test failed: " << error.what() << '\n';
        return 1;
    }
}
