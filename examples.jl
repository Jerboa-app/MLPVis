using Flux, PerceptualColourMaps, ProgressMeter, Distributions, ArgParse

# change to CairoMakie if without gpu
# tried an if-else but that borked my Julia for some reason...
# (ERROR: LoadError: InitError: could not load library libharfbuzz-gobject.so")
using GLMakie

s = ArgParseSettings()

@add_arg_table! s begin
    "--resolution-upscaling"
    help = "Modifier to 1920x1080:\n -2 = 480x270,\n -1 = 960x540,\n 1 = 1920x1080 (DEFAULT),\n 2 = 3840x2160"
    arg_type = Int
    default = 1
end

Arguments = parse_args(s)

@info Arguments

scale = Arguments["resolution-upscaling"]
res = (1920,1080)
if res == -2
    res = (480,270)
elseif res == -1
    res = (960,540)
elseif res == 2
    res = (3840,2160)
end

include("visualisation.jl")

# comment out these two lines to watch live, it's much faster without though
set_window_config!(framerate = Inf, vsync = false)
inline!(true)

fps = 5
# 1D

# no noise 1d linear

model = Chain(Dense(1,3),Dense(3,3),Dense(3,1))

for l in model.layers
    l.W .= .1
    l.b .= 0.0
end

f(x) = 5.0*x

x = collect(0.0:0.01:1.0)
y = f.(x);
pred = [model([i]) for i in x]

params_ = Flux.params(model);

Data = [([x[i]],[y[i]]) for i in 1:length(x)];

loss(model,x,y) = sum((model(x).-y).^2.0)

WatchNetwork(model,Data,loss,y,fps=fps,Epochs=1000,
res=(1920*2,1080*2),
filename="linear.mp4",opt=Descent(0.0001))

# Cauchy + normal noise in 1D

model = Chain(Dense(1,3),Dense(3,3),Dense(3,1))

for l in model.layers
    l.W .= .1
    l.b .= 0.
end
η=0.25
f(x) = 5.0*x+(rand(Cauchy(0,η))%5)+randn()*η

x = collect(0.0:0.01:1.0)
y = f.(x);
pred = [model([i]) for i in x]

loss(x,y) = sum((model(x).-y).^2.0)
params_ = Flux.params(model);

Data = [([x[i]],[y[i]]) for i in 1:length(x)];
cb = () -> @show sum([loss(x,y) for (x,y) in Data])/length(Data)

loss(model,x,y) = sum((model(x).-y).^2.0)

WatchNetwork(model,Data,loss,y,fps=fps,Epochs=600,res=(1920*2,1080*2), opt=Descent(0.001))

# 2D

# linear problem

model = Chain(Dense(2,2),Dense(2,2))
f(x,y)= 0.4*x < y

N = 32
x = range(0,stop=1.,length=N)
y = copy(x)
t = [f(i,j)==1 ? [0,1] : [1,0] for i in x for j in y];
pred = [model([i,j]) for i in x for j in y]

loss(x,y) = Flux.Losses.logitbinarycrossentropy(softmax(model(x)),y)
params_ = Flux.params(model);

Data = [([i,j],f(i,j)==1 ? [0,1] : [1,0]) for i in x for j in y];

n = [d==[0,1] for d in map(x->x[2],Data)]
m = [d==[1,0] for d in map(x->x[2],Data)]

Data_old = copy(Data)

if sum(n) < sum(m)
    for i in 1:abs(sum(n) - sum(m))
        push!(Data,rand(Data_old[n]))
    end
else
    for i in 1:abs(sum(n) - sum(m))
        push!(Data,rand(Data_old[m]))
    end
end
n = [d==[0,1] for d in map(x->x[2],Data)]
m = [d==[1,0] for d in map(x->x[2],Data)]
sum(n),sum(m)

loss(model,x,y) = Flux.Losses.logitbinarycrossentropy(softmax(model(x)),y)

WatchNetwork(model,Data,loss,t,fps=5,Epochs=2000,res=(1920*2,1080*2),
filename="2d-linear.mp4",opt=Descent(0.005))

# nonlinear with linear network

model = Chain(Dense(2,2),Dense(2,2),Dense(2,2))

f(x,y)= x<y || x^2+y^2 < 0.5
N = 32
x = range(0,stop=1.,length=N)
y = copy(x)
t = [f(i,j)==1 ? [0,1] : [1,0] for i in x for j in y];
pred = [model([i,j]) for i in x for j in y]

loss(x,y) = Flux.Losses.logitbinarycrossentropy(softmax(model(x)),y)
params_ = Flux.params(model);

Data = [([i,j],f(i,j)==1 ? [0,1] : [1,0]) for i in x for j in y];

n = [d==[0,1] for d in map(x->x[2],Data)]
m = [d==[1,0] for d in map(x->x[2],Data)]

Data_old = copy(Data)

if sum(n) < sum(m)
    for i in 1:abs(sum(n) - sum(m))
        push!(Data,rand(Data_old[n]))
    end
else
    for i in 1:abs(sum(n) - sum(m))
        push!(Data,rand(Data_old[m]))
    end
end
n = [d==[0,1] for d in map(x->x[2],Data)]
m = [d==[1,0] for d in map(x->x[2],Data)]
sum(n),sum(m)

loss(model,x,y) = Flux.Losses.logitbinarycrossentropy(softmax(model(x)),y)

WatchNetwork(model,Data,loss,t,fps=5,Epochs=1000,res=(1920*2,1080*2),
filename="2d-nonlinear-bad.mp4",opt=Descent(0.001))

# non-linear problem with activations

model = Chain(Dense(2,2,tanh),Dense(2,2,tanh),Dense(2,2,tanh))

f(x,y)= x<y || x^2+y^2 < 0.5
N = 32
x = range(0,stop=1.,length=N)
y = copy(x)
t = [f(i,j)==1 ? [0,1] : [1,0] for i in x for j in y];
pred = [model([i,j]) for i in x for j in y]

loss(x,y) = Flux.Losses.logitbinarycrossentropy(softmax(model(x)),y)
params_ = Flux.params(model);

Data = [([i,j],f(i,j)==1 ? [0,1] : [1,0]) for i in x for j in y];

n = [d==[0,1] for d in map(x->x[2],Data)]
m = [d==[1,0] for d in map(x->x[2],Data)]

Data_old = copy(Data)

if sum(n) < sum(m)
    for i in 1:abs(sum(n) - sum(m))
        push!(Data,rand(Data_old[n]))
    end
else
    for i in 1:abs(sum(n) - sum(m))
        push!(Data,rand(Data_old[m]))
    end
end
n = [d==[0,1] for d in map(x->x[2],Data)]
m = [d==[1,0] for d in map(x->x[2],Data)]
sum(n),sum(m)

loss(model,x,y) = Flux.Losses.logitbinarycrossentropy(softmax(model(x)),y)

WatchNetwork(model,Data,loss,t,fps=5,Epochs=3000,res=(1920*2,1080*2),
filename="2d-nonlinear.mp4",opt=Descent(0.001),neuron_colour=RGBAf0(0.6,0.6,0,1))

# Polkadots!

model = Chain(Dense(2,4,tanh),Dense(4,8,tanh),Dense(8,8,tanh),Dense(8,4,tanh),Dense(4,2,tanh))

f(x,y)= x^2+y^2 < 0.1 || (x-0.6)^2+(y-0.5)^2 < 0.1
N = 32
x = range(0,stop=1.,length=N)
y = copy(x)
t = [f(i,j)==1 ? [0,1] : [1,0] for i in x for j in y];
pred = [model([i,j]) for i in x for j in y]

loss(x,y) = Flux.Losses.logitbinarycrossentropy(softmax(model(x)),y)
params_ = Flux.params(model);

Data = [([i,j],f(i,j)==1 ? [0,1] : [1,0]) for i in x for j in y];

n = [d==[0,1] for d in map(x->x[2],Data)]
m = [d==[1,0] for d in map(x->x[2],Data)]

Data_old = copy(Data)

if sum(n) < sum(m)
    for i in 1:abs(sum(n) - sum(m))
        push!(Data,rand(Data_old[n]))
    end
else
    for i in 1:abs(sum(n) - sum(m))
        push!(Data,rand(Data_old[m]))
    end
end
n = [d==[0,1] for d in map(x->x[2],Data)]
m = [d==[1,0] for d in map(x->x[2],Data)]
sum(n),sum(m)

loss(model,x,y) = Flux.Losses.logitbinarycrossentropy(softmax(model(x)),y)

WatchNetwork(model,Data,loss,t,fps=5,Epochs=6000,res=(1920*2,1080*2),
filename="2d-polkadot.mp4",opt=ADAGrad(0.01),neuron_colour=RGBAf0(0.6,0.6,0,1))

# Some sort of triangle thing

model = Chain(Dense(2,4,tanh),Dense(4,8,tanh),Dense(8,8,tanh),Dense(8,4,tanh),Dense(4,2,tanh))

f(x,y)= 0.4*x < y && 0.4*y < x
N = 32
x = range(0,stop=1.,length=N)
y = copy(x)
t = [f(i,j)==1 ? [0,1] : [1,0] for i in x for j in y];
pred = [model([i,j]) for i in x for j in y]

loss(x,y) = Flux.Losses.logitbinarycrossentropy(softmax(model(x)),y)
params_ = Flux.params(model);

Data = [([i,j],f(i,j)==1 ? [0,1] : [1,0]) for i in x for j in y];

n = [d==[0,1] for d in map(x->x[2],Data)]
m = [d==[1,0] for d in map(x->x[2],Data)]

Data_old = copy(Data)

if sum(n) < sum(m)
    for i in 1:abs(sum(n) - sum(m))
        push!(Data,rand(Data_old[n]))
    end
else
    for i in 1:abs(sum(n) - sum(m))
        push!(Data,rand(Data_old[m]))
    end
end
n = [d==[0,1] for d in map(x->x[2],Data)]
m = [d==[1,0] for d in map(x->x[2],Data)]
sum(n),sum(m)

loss(model,x,y) = Flux.Losses.logitbinarycrossentropy(softmax(model(x)),y)

WatchNetwork(model,Data,loss,t,fps=5,Epochs=6000,res=(1920*2,1080*2),
filename="2d-triangle.mp4",opt=ADAGrad(0.001),neuron_colour=RGBAf0(0.6,0.6,0,1))
