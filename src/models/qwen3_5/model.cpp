#include "models/qwen3_5/model.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing, std::vector<std::vector<float>> hadamard_tables)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), resources_(std::move(resources)),
      info_(std::move(info)) {
    // Upload the shared folded-ternary sign tables once; the per-use views below stay valid for
    // the model's lifetime because the buffer outlives every borrower.
    std::size_t total = 0;
    for (const auto& table : hadamard_tables) { total += table.size(); }
    if (total > 0) {
        hadamard_device_ = DeviceBuffer(total * sizeof(float));
        std::size_t offset = 0;
        for (auto& table : hadamard_tables) {
            hadamard_device_.copy_from_host(table.data(), table.size() * sizeof(float),
                                             offset * sizeof(float));
            hadamard_ptrs_.push_back(
                {static_cast<const float*>(hadamard_device_.p) + offset,
                 static_cast<std::int32_t>(table.size() / 1024)});
            offset += table.size();
        }
    }
}

Model::~Model() = default;

ops::WeightInput Model::input(WeightUseId id) const {
    const auto& parameter = weight(id.parameter);
    const auto& use       = parameter.uses.at(id.use_index);
    auto in              = ops::WeightInput{parameter.view, use.policy, use.activation_input_divisor};
    if (use.hadamard_table >= 0) {
        const auto& [signs, blocks] = hadamard_ptrs_.at(static_cast<std::size_t>(use.hadamard_table));
        in.hadamard_signs            = signs;
        in.hadamard_n_blk            = blocks;
    }
    return in;
}

ops::WeightInput Model::input(WeightId id) const {
    if (weight(id).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0});
}

} // namespace ninfer::models::qwen3_5
