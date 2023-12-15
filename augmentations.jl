struct RandomTranslate{N} <: DataAugmentation.ProjectiveTransform
    shift::NTuple{N, Int}
end
RandomTranslate(sz, shift::NTuple{2, <:AbstractFloat}) =
    RandomTranslate(map((s, scale) -> round(Int, s * scale), sz, shift))

DataAugmentation.getrandstate(tfm::RandomTranslate) = map(s -> rand(-s:s), tfm.shift)

DataAugmentation.getprojection(tfm::RandomTranslate, bounds::Bounds;
                               randstate = DataAugmentation.getrandstate(tfm)) =
    Translation(randstate...)

apply_augmenation(pipeline, x) =
    itemdata(DataAugmentation.apply(pipeline, Image(x)))
apply_augmenation(pipeline, x::AbstractArray{<:RGB, 3}) =
    cat(map(Base.Fix1(apply_augmenation, pipeline), eachslice(x; dims = 3))...;
        dims = 4)
apply_augmenation(pipeline, x::NamedTuple{(:features, :targets)}) =
    (image = apply_augmenation(pipeline, x.features), label = x.targets)
map_augmentation(pipeline, data) =
    MLUtils.mapobs(Base.Fix1(apply_augmenation, pipeline), data)
