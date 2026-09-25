#include "models/qwen3_5/load/bindings.h"

#include "artifact/views.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ninfer::models::qwen3_5::loading {

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format) {
    if (parameters_.contains(name)) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
    PendingWeight pending;
    pending.reference =
        binder.parameter(name, std::move(shape), artifact::Residency::Device, exact_format);
    for (const auto& input : inputs) {
        const auto& use = binder.use(name, input);
        if (!use.activation_policy) {
            throw artifact::ArtifactError(name + "@" + input + ": missing activation policy");
        }
        WeightUse result;
        result.input = input;
        switch (*use.activation_policy) {
        case artifact::ActivationPolicy::A16Only:
            result.policy = ops::LinearPolicy::A16Only;
            break;
        case artifact::ActivationPolicy::AllowA8:
            result.policy = ops::LinearPolicy::AllowA8;
            break;
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::AllowA4;
            break;
        }
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role == "activation_input_divisor") {
                const auto value = binder.values(binding, QType::FP32).scalar_f32();
                if (!std::isfinite(value) || value <= 0) {
                    throw artifact::ArtifactError(name + "@" + input +
                                                  ": activation divisor must be positive finite FP32");
                }
                result.activation_input_divisor = value;
            } else if (role == "hadamard_signs") {
                // The artifact stores the folded-ternary sign rows as a BF16 block of
                // 1024-element rows; the rotation kernels consume them as FP32, so convert
                // once here and share the table by element count (widths never repeat).
                const auto values = binder.values(binding, QType::BF16);
                if (values.elements == 0 || values.elements % 1024 != 0) {
                    throw artifact::ArtifactError(
                        name + "@" + input + ": hadamard sign block must be a nonzero multiple of 1024");
                }
                std::int32_t index = -1;
                for (std::size_t table = 0; table < hadamard_tables.size(); ++table) {
                    if (hadamard_tables[table].size() == values.elements) { index = static_cast<std::int32_t>(table); break; }
                }
                if (index < 0) {
                    hadamard_tables.emplace_back(values.elements);
                    auto& target = hadamard_tables.back();
                    for (std::uint64_t i = 0; i < values.elements; ++i) {
                        std::uint16_t bits;
                        std::memcpy(&bits, values.data.data() + i * 2, sizeof(bits));
                        target[i] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
                    }
                    index = static_cast<std::int32_t>(hadamard_tables.size() - 1);
                }
                result.hadamard_table = index;
            } else {
                throw artifact::ArtifactError(name + "@" + input + ": unknown auxiliary " + role);
            }
        }
        pending.uses.push_back(std::move(result));
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().object(part.object)));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, QType format) {
    return parameter(std::move(name), std::move(shape), {}, format);
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (auto& item : pending) {
        auto view = artifact::bind_view(item.reference, materialized);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses)});
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
