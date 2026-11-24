#include "conditioning/conditioning_cache.h"
#include "pipeline/diffusion_engine.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
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
    std::vector<float> output = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};

    SDCondition get_learned_condition(int, const ConditionerParams&) override {
        ++calls;
        SDCondition condition;
        condition.c_crossattn = sd::Tensor<float>({2, 3}, output);
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
            TemporaryDirectory invalid_temp;
            FakeConditioner faulty;
            faulty.output.assign(6, 0.f);
            ConditionerParams invalid_params = params;
            invalid_params.text = "invalid moss encoding";
            {
                ConditioningCache old_cache;
                old_cache.configure_persistence(invalid_temp.path.string(), "moss validation");
                old_cache.get(faulty, 1, invalid_params);
                require(old_cache.size() == 1, "setup must persist legacy invalid entry");
            }
            ConditioningCache cache;
            cache.configure_persistence(invalid_temp.path.string(), "moss validation");
            cache.set_moss_validation(true);
            require(!cache.contains(invalid_params), "invalid legacy encoding must not hit");
            require(cache.get(faulty, 1, invalid_params).empty(), "all-zero encoding must fail");
            require(cache.size() == 0, "invalid encoding must not be cached");
            faulty.output = {1.f, 2.f, std::numeric_limits<float>::quiet_NaN(), 4.f, 5.f, 6.f};
            require(cache.get(faulty, 1, invalid_params).empty(), "non-finite encoding must fail");
            faulty.output = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
            require(!cache.get(faulty, 1, invalid_params).empty(), "valid encoding must succeed");
            const int calls = faulty.calls;
            cache.get(faulty, 1, invalid_params);
            require(faulty.calls == calls, "valid encoding must be reused");
            cache.set_capacity(0);
            faulty.output.assign(6, 0.f);
            require(cache.get(faulty, 1, invalid_params).empty(), "disabled cache must still validate");
        }

        {
            StableDiffusionGGML engine;
            auto faulty = std::make_shared<FakeConditioner>();
            faulty->output.assign(6, 0.f);
            engine.cond_stage_model = faulty;
            engine.conditioning_cache_->set_moss_validation(true);
            std::string detail;
            engine.generation_progress.set_callback(
                [](const sd_generation_progress_t* progress, void* data) {
                    *static_cast<std::string*>(data) = progress->detail ? progress->detail : "";
                }, &detail);
            require(engine.get_learned_condition(params).empty(), "engine must reject invalid encoding");
            require(detail == "INVALID_MOSS_TEXT_ENCODING", "engine must report invalid encoding");
        }

        {
            ConditioningCache cache;
            cache.set_capacity(2);
            cache.configure_persistence(temp.path.string(), "model scope");
            const auto result = cache.get(conditioner, 1, params);
            require(conditioner.calls == 1, "cache miss should encode once");
            require(result.c_crossattn.values() == std::vector<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}),
                    "fresh conditioning mismatch");
            cache.get(conditioner, 1, params);
            require(conditioner.calls == 1, "same conditioning params should not encode twice");
            cache.record_role(params, 1);
            cache.record_role(params, 2);
            const auto id = cache.list().front().id;
            std::ifstream roles(temp.path / cache.storage_scope_id() / (id + ".roles"));
            int value = 0;
            roles >> value;
            require(value == 3, "positive and negative roles should both be persisted");
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
            for (const auto& item : std::filesystem::directory_iterator(cache_dir)) {
                require(item.path().extension() != ".roles", "clear should delete prompt role metadata");
            }
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
            StableDiffusionGGML engine;
            engine.conditioning_cache_directory_ = temp.path.string();
            engine.conditioning_cache_model_scope_ = "runner reset model scope";
            engine.conditioning_cache_->set_capacity(2);
            engine.configure_conditioning_cache_for_loras({});
            engine.conditioning_cache_->get(conditioner, 1, params);
            const int calls_before_reset = conditioner.calls;
            require(engine.reset_runners({}), "empty runner reset should succeed");
            engine.configure_conditioning_cache_for_loras({});
            require(engine.conditioning_cache_->contains(params),
                    "runner reset must preserve persisted conditioning");
            engine.conditioning_cache_->get(conditioner, 1, params);
            require(conditioner.calls == calls_before_reset,
                    "runner reset should restore conditioning without encoding");
        }

        {
            const std::string lora_a_scope = "model scope;lora=variant-a";
            const std::string lora_b_scope = "model scope;lora=variant-b";
            {
                ConditioningCache lora_a;
                lora_a.set_capacity(2);
                lora_a.configure_persistence(temp.path.string(), lora_a_scope);
                lora_a.get(conditioner, 1, params);
            }
            const int calls_after_lora_a = conditioner.calls;
            {
                ConditioningCache lora_b;
                lora_b.set_capacity(2);
                lora_b.configure_persistence(temp.path.string(), lora_b_scope);
                require(lora_b.size() == 0, "different LoRA scope must not reuse entries");
            }
            {
                ConditioningCache restored_lora_a;
                restored_lora_a.set_capacity(2);
                restored_lora_a.configure_persistence(temp.path.string(), lora_a_scope);
                require(restored_lora_a.contains(params),
                        "same LoRA scope should restore its persisted entry");
                restored_lora_a.get(conditioner, 1, params);
                require(conditioner.calls == calls_after_lora_a,
                        "restored LoRA conditioning should not rerun the encoder");
            }
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

        {
            TemporaryDirectory prompt_temp;
            const std::string scope = "prompt image scope";
            sd::PromptInput input_a = {
                {sd::PromptPart::Type::TEXT, params.text},
                {sd::PromptPart::Type::IMAGE, "image-a", std::make_shared<const sd::Tensor<float>>(sd::Tensor<float>({1, 1, 1, 1}, {0.25f}))},
            };
            sd::PromptInput input_b       = input_a;
            input_b[1].image              = std::make_shared<const sd::Tensor<float>>(sd::Tensor<float>({1, 1, 1, 1}, {0.75f}));
            auto image_params             = params;
            image_params.prompt_input     = &input_a;
            const int calls_before_images = conditioner.calls;

            {
                ConditioningCache cache;
                cache.set_capacity(4);
                cache.configure_persistence(prompt_temp.path.string(), scope);
                const auto first = cache.get(conditioner, 1, image_params);
                require(conditioner.calls == calls_before_images + 1, "first prompt image should encode once");
                cache.get(conditioner, 1, image_params);
                require(conditioner.calls == calls_before_images + 1, "prompt image should hit in memory");

                conditioner.output        = {7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
                image_params.prompt_input = &input_b;
                const auto second         = cache.get(conditioner, 1, image_params);
                require(conditioner.calls == calls_before_images + 2, "different prompt image should encode separately");
                require(first.c_crossattn.values() != second.c_crossattn.values(), "different images must not collide in memory");
                cache.get(conditioner, 1, image_params);
                require(conditioner.calls == calls_before_images + 2, "second prompt image should hit in memory");
                require(cache.size() == 2, "prompt images should remain cached in memory");
                cache.record_role(image_params, 1);
                require(std::filesystem::is_empty(prompt_temp.path / cache.storage_scope_id()),
                        "prompt images must not write cache or role files");
            }

            ConditioningCache restored;
            restored.set_capacity(4);
            restored.configure_persistence(prompt_temp.path.string(), scope);
            require(restored.size() == 0, "prompt images must not be restored from disk");
            require(!restored.contains(params), "text-only params must not hit a persisted prompt image");
            image_params.prompt_input = &input_a;
            require(!restored.contains(image_params), "first prompt image must miss after reload");
            image_params.prompt_input = &input_b;
            require(!restored.contains(image_params), "second prompt image must miss after reload");

            conditioner.output   = {13.f, 14.f, 15.f, 16.f, 17.f, 18.f};
            const auto text_only = restored.get(conditioner, 1, params);
            require(conditioner.calls == calls_before_images + 3, "text-only request must encode after prompt image reload");
            require(text_only.c_crossattn.values() == conditioner.output, "text-only request must not reuse image conditioning");
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "conditioning-cache test failed: " << error.what() << '\n';
        return 1;
    }
}
