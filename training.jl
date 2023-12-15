function train(model, loaders, opt; logger, nepochs = 50)
    trainloader, testloader = loaders
    for epoch in 1:nepochs
        loss = 0f0
        acc = 0f0
        nsamples = 0
        for (x, y) in trainloader
            x, y = gpu(x), gpu(y)
            _loss, grad = Flux.withgradient(model) do m
                Flux.logitcrossentropy(m(x), y)
            end
            opt, model = Flux.update!(opt, model, grad[1])
            loss += _loss
            acc += sum(Flux.onecold(m(x)) .== Flux.onecold(y))
            nsamples += size(x)[end]
        end
        trainloss = loss / nsamples
        trainacc = acc / nsamples

        loss = 0f0
        acc = 0f0
        nsamples = 0
        for (x, y) in testloader
            x, y = gpu(x), gpu(y)
            ŷ = model(x)
            loss += Flux.logitcrossentropy(ŷ, y)
            acc += sum(Flux.onecold(ŷ) .== Flux.onecold(y))
            nsamples += size(x)[end]
        end
        testloss = loss / nsamples
        testacc = acc / nsamples

        Wandb.log(logger, Dict("epoch" => epoch,
                               "train/accuracy" => trainacc,
                               "train/loss" => trainloss,
                               "test/accuracy" => testacc,
                               "test/loss" => testloss))
    end
end
