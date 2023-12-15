using Metalhead
using Flux
using MLDatasets: CIFAR10, convert2image
using MLUtils
# using CoordinateTransformations
using DataAugmentation
using CUDA
using Wandb
using Dates: now
using Images

include("training.jl")
include("augmentations.jl")

## load the data

nclasses = length(CIFAR10().metadata["class_names"])
traindata = CIFAR10(; split = :train)[:]
traindata = (features = convert2image(CIFAR10, traindata.features),
             targets = Flux.onehotbatch(traindata.targets, 0:(nclasses - 1)))
testdata = CIFAR10(; split = :test)[:]
testdata = (features = convert2image(CIFAR10, testdata.features),
            targets = Flux.onehotbatch(testdata.targets, 0:(nclasses - 1)))
;

##

augmentations = Rotate(10) |>
                Zoom((0.9, 1.1)) |>
                ScaleFixed((32, 32)) |>
                Maybe(FlipX()) |>
                CenterCrop((32, 32)) |>
                ImageToTensor()
trainaug = map_augmentation(augmentations, traindata)
testaug = map_augmentation(ImageToTensor(), testdata)
;

##

bs = 128
trainloader = DataLoader(trainaug;
                         batchsize = bs,
                         shuffle = true,
                         buffer = true,
                         parallel = true)
testloader = DataLoader(testaug; batchsize = bs, buffer = true)

##

MODELS = Dict(
    :EfficientNet => [
        (:b0,),
        (:b1,),
        (:b2,),
        (:b3,),
        (:b4,),
        (:b5,),
        (:b6,),
        (:b7,),
        (:b8,)
    ],
    :ResNet => [
        (18,),
        (34,),
        (50,),
        (101,),
        (152,)
    ]
)

##

for (model, cfgs) in pairs(MODELS)
    for cfg in cfgs
        @eval m = $model($cfg...; nclasses = 10) |> gpu
        opt = Flux.setup(AdamW(), m)
        logger = WandbLogger(project = "metalhead-bench",
                             name = "$model-$cfg-$(now())")
        train(m, (trainloader, testloader), opt; logger)
        close(logger)
        GC.gc(true)
    end
end
