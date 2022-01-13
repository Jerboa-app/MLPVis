function maptocolours(x,cmap)
    m=minimum(x)
    M=maximum(x)
    if m == M
        return [cmap[128] for i in 1:length(x)]
    end
    y = Int.(floor.(255*(x.-m)/(M-m)))
    return [cmap[i+1] for i in y]
end

"""
    Train model on data, against the loss. Plots the network (nodes/edges) with colours
    representing the relative connection strength (blue = negative, white = 0, red = positive)
    compared to all others (biases and weights done separately).

    Plots the current performance on data as well.

    Saves an mp4 of the process.
"""
function WatchNetwork(model::Flux.Chain,data, loss::Function, truth; Epochs = 100, opt=Descent(0.01),
        debug=false,minwidth=1.0,maxwidth=4.0,res=(1920,1080),fig=nothing, neuron_colour=RGBAf0(0,0,0,1),
        bias_colour=RGBAf0(0.5,0.5,0.5,1),fps=60,titlesize=30,filename="Makie.mp4"
)::GLMakie.Figure

    fig = GLMakie.Figure(resolution=res,fontsize=titlesize)

    main_title = Node("Network") # will update to track epochs
    axes = [GLMakie.Axis(fig[1,1],title=main_title)]

    title = Node("")

    if data != nothing
        push!(axes,GLMakie.Axis(fig[1,2],title=title))
    end

    time_ = Node(1)

    Neurons = Vector{Point2{Float64}}()
    Biases = Vector{Point2{Float64}}()
    Connections = Vector{Point2{Float64}}()
    ConnectionStrength = Vector{Float64}()
    nColours = Vector{PerceptualColourMaps.RGBA{Float64}}()
    bColours = Vector{PerceptualColourMaps.RGBA{Float64}}()

    cm = cmap("D1")

    for layer in model.layers
        if !(typeof(layer)<:Flux.Dense)
            @warn "Model must only have Dense layers"
            @assert typeof(layer)<:Flux.Dense
        end
    end

    Ns = [(size(layer.weight,2),size(layer.weight,1)) for layer in model.layers]
    biases = true

    spacing_x = 8.0
    spacing_y = 4.0
    neuron_size = 1.0
    bias_size = 0.5
    bias_placement = [0*bias_size,+2*bias_size] # gets added to neuron position

    pos = [(1,1)]

    # maximum bias magnitude
    B = 0.0
    for l in model.layers
        for b in abs.(l.bias)
            if b > B
                B = b
            end
        end
    end

    # collect neurons and biases into arrays
    for l in 1:length(Ns)
        start = 0.0
        ns = Ns[l][1]
        if ns > 1
            if ns % 2 == 0
                start = -spacing_y * (ns / 2)+spacing_y/2.
            else
                start = -spacing_y * floor(ns / 2)
            end
        end
        for n in 1:Ns[l][1]
            # add a neuron
            x,y = spacing_x*(l-1),start+spacing_y*(n-1) # with padding
            neuron = Point2(x,y)
            push!(Neurons,neuron)
            push!(nColours,neuron_colour)
        end

        start = 0.0
        ns = Ns[l][2]
        if ns > 1
            if ns % 2 == 0
                start = -spacing_y * (ns / 2)+spacing_y/2.
            else
                start = -spacing_y * floor(ns / 2)
            end
        end
        if biases
            for n in 1:Ns[l][2]
                x,y = spacing_x*(l),start+spacing_y*(n-1)
                # add it's bias
                bias = Point2(x+bias_placement[1],y+bias_placement[2])
                push!(Biases,bias)
                push!(bColours,bias_colour)
                # add a connection for the bias
                push!(Connections,Point2(x,y))
                push!(Connections,Point2(x+bias_placement[1],y+bias_placement[2]))
                # relative magnitude*sign
                mag = sign(model.layers[l].bias[n])*maxwidth*(minwidth+abs.(model.layers[l].bias[n])/B)
                isnan(mag) ? mag = 0 : nothing
                push!(ConnectionStrength,mag)
                push!(ConnectionStrength,mag)
            end
        end

        # final layer
        if l == length(Ns)
            # draw output
            start = 0.0
            ns = Ns[l][2]
            if ns > 1
                if ns % 2 == 0
                    start = -spacing_y * (ns / 2)+spacing_y/2.
                else
                    start = -spacing_y * floor(ns / 2)
                end
            end
            for n in 1:Ns[l][2]
                # add a neuron
                x,y = spacing_x*(l),start+spacing_y*(n-1)

                neuron = Point2(x,y)
                push!(Neurons,neuron)
                push!(nColours,neuron_colour)
            end
        end

        # largest weight magnitude
        M = 0.0
        for l in model.layers
            for w in abs.(l.W)
                if w > M
                    M = w
                end
            end
        end

        # draw connections
        for n1 in 1:Ns[l][1]
            for n2 in 1: Ns[l][2]
                start1 = 0.0
                start2 = 0.0
                ns1 = Ns[l][1]
                if ns1 > 1
                    if ns1 % 2 == 0
                        start1 = -spacing_y * (ns1 / 2)+spacing_y/2.
                    else
                        start1 = -spacing_y * floor(ns1 / 2)
                    end
                end

                ns2 = Ns[l][2]
                if ns2 > 1
                    if ns2 % 2 == 0
                        start2 = -spacing_y * (ns2 / 2)+spacing_y/2.
                    else
                        start2 = -spacing_y * floor(ns2 / 2)
                    end
                end

                x1,y1 = spacing_x*(l-1),start1+spacing_y*(n1-1)
                x2,y2 = spacing_x*(l),start2+spacing_y*(n2-1)
                push!(Connections,Point2(x1,y1))
                push!(Connections,Point2(x2,y2))
                # relative magnitude*sign
                mag = sign(model.layers[l].weight[n2,n1]/M)*maxwidth*(minwidth+abs.(model.layers[l].weight[n2,n1])/M)
                isnan(mag) ? mag = 0 : nothing
                push!(ConnectionStrength,mag)
                push!(ConnectionStrength,mag)
            end
        end
    end

    # make the arrays into nodes so we can update them
    Neurons = Node(Neurons)
    Biases = Node(Biases)
    Connections = Node(Connections)
    ConnectionStrength = Node(abs.(ConnectionStrength))
    ConnectionColour = Node(maptocolours(ConnectionStrength[],cm))
    nColours = Node(nColours);
    bColours = Node(bColours);

    @show ConnectionStrength[]

    # plot neurons, biases, and weights
    GLMakie.scatter!(axes[1],Neurons,color=nColours,markersize=(neuron_size,neuron_size),markerspace=SceneSpace)
    GLMakie.scatter!(axes[1],Biases,color=bColours,markersize=(bias_size,bias_size),markerspace=SceneSpace)
    GLMakie.linesegments!(axes[1],Connections,linewidth=ConnectionStrength,color=ConnectionColour)

    ns = maximum([max(l[1],l[2]) for l in Ns])

    l = length(model.layers)

    # determine position of viewport
    # so the network is drawn centered

    mx = 0+bias_placement[1]-bias_size
    Mx = l*spacing_x + neuron_size

    my = 0
    My = 0

    if ns % 2 == 0
        my = -spacing_y * (0.5+ns / 2)-bias_placement[2] - bias_size
        My = spacing_y * (ns / 2)+ bias_placement[2] + bias_size
    else
        my = -spacing_y * floor(ns / 2) - bias_placement[2] - bias_size
        My = spacing_y * floor(ns / 2) + bias_placement[2] + bias_size
    end

    dx = Mx-mx
    dy = My-my

    if dx < dy
        Mx = Mx+(dy-dx)
    elseif dy < dx
        My = My +(dx-dy)
    end

    # re-center
    pad = floor(l/2.)*spacing_x
    if l % 2 != 0
        pad += spacing_x/2.0
    end

    c = ((Mx+mx)/2.0-pad,(My+my)/2.0)
    # set viewport
    axes[1].aspect = AxisAspect(1)
    limits!(axes[1],mx-c[1],Mx-c[1],my-c[2],My-c[2])
    if debug == false
        hidedecorations!(axes[1,1])
        hidespines!(axes[1,1])
    end
    if length(data[1][1])==1 && length(data[1][2])==1
        # 1-d problem plotting code
        p = [model(data[i][1])[1] for i in 1:length(data)]|>Node
        if data != nothing
            x = [data[i][1][1] for i in 1:length(data)]
            GLMakie.scatter!(axes[2],x,truth)
            GLMakie.lines!(axes[2],x,p,linewidth=2.0,color="red")
            e = sum([loss(model,data[i][1],data[i][2]) for i in 1:length(data)])/length(data)
            title[]="Error: $(round(e,digits=5))"
        end
    elseif length(data[1][1])==2 && length(data[1][2])==2
        # 2-d problem plotting code
        p=[model(data[i][1]) for i in 1:length(data)]|>Node
        if data != nothing
            axes[2].aspect = AxisAspect(1)
            hidedecorations!(axes[2])
            N = length(truth)|>floor|>Int
            n = sqrt(N)|>floor|>Int

            c = reshape([RGBAf0(x[1],x[2],0,1) for x in truth],n,n)
            pos = Vector{Point2}()
            col = Vector{RGBAf0}()
            for i in 1:size(c,1)
                for j in 1:size(c,2)
                    push!(pos,Point2(i-0.5,j-0.5))
                    push!(col,c[i,j])
                end
            end
            push!(axes,GLMakie.Axis(
                fig[1,3],title="Goal"
            ))
            axes[3].aspect = AxisAspect(1)
            hidedecorations!(axes[3])
            GLMakie.scatter!(axes[3],pos,marker=Rect2D(0,0,1,1), # like a heatmap
                    markersize=(1,1),markerspace=SceneSpace,
                    color = col
            )

            q = reshape([RGBAf0(x[1],x[2],0,1) for x in p[][1:N]],n,n)|>Node
            pos = Vector{Point2}()
            col = Vector{RGBAf0}()
            for i in 1:size(q[],1)
                for j in 1:size(q[],2)
                    push!(pos,Point2(i-0.5,j-0.5))
                    push!(col,q[][i,j])
                end
            end
            pos=Node(pos)
            col = Node(col)
            GLMakie.scatter!(axes[2],pos,marker=Rect2D(0,0,1,1),
                markersize=(1,1),markerspace=SceneSpace,
                color = col
            )
            e = sum([loss(model,data[i][1],data[i][2]) for i in 1:length(data)])/length(data)
            a = Flux.Losses.logitbinarycrossentropy([0,1],[1,0])
            e = (e-a) # normalise loss
            title[]="Error: $(round(e,digits=5))"
        end
    end

    prog = Progress(Epochs-1)

    # now begin training and recording

    GLMakie.record(fig, filename, collect(2:Epochs), framerate=fps) do k

        # train 1 epoch
        Flux.train!((x,y)->loss(model,x,y),Flux.params(model),data,opt)

        newConnectionStrength = []

        if data != nothing
            main_title[] = "Network (Epochs: $k)"
            if length(data[1][1])==1 && length(data[1][2])==1
                # update prediction node
                p[] = [model(data[i][1])[1] for i in 1:length(data)]
                e = sum([loss(model,data[i][1],data[i][2]) for i in 1:length(data)])/length(data)
                # title node with error
                title[] = "Error: $(round(e,digits=5))"
            elseif length(data[1][1])==2 && length(data[1][2])==2
                # update prediction node
                p[]=[model(data[i][1]) for i in 1:length(data)]
                N = length(truth)|>floor|>Int
                n = sqrt(N)|>floor|>Int
                q[] = reshape([RGBAf0(x[1],x[2],0,1) for x in p[][1:N]],n,n)
                newpos = []
                newcol = []
                for i in 1:size(q[],1)
                    for j in 1:size(q[],2)
                        push!(newpos,Point2(i-0.5,j-0.5))
                        push!(newcol,q[][i,j])
                    end
                end
                # update current performance node
                pos[] = newpos
                col[] = newcol
                e = sum([loss(model,data[i][1],data[i][2]) for i in 1:length(data)])/length(data)
                a = Flux.Losses.logitbinarycrossentropy([0,1],[0,1])
                e = (e-a)
                title[]="Error: $(round(e,digits=5))"
            end
        end

        # update colour nodes
        B = 0.0
        for l in model.layers
            for b in abs.(l.bias)
                if b > B
                    B = b
                end
            end
        end

        for l in 1:length(Ns)
            start = 0.0
            ns = Ns[l][1]

            if biases
                for n in 1:Ns[l][2]
                    mag = sign(model.layers[l].bias[n])*maxwidth*(minwidth+abs.(model.layers[l].bias[n])/B)
                    isnan(mag) ? mag = 0 : nothing
                    push!(newConnectionStrength,mag)
                    push!(newConnectionStrength,mag)
                end
            end

            M = 0.0
            for l in model.layers
                for w in abs.(l.W)
                    if w > M
                        M = w
                    end
                end
            end

            # draw connections
            for n1 in 1:Ns[l][1]
                for n2 in 1: Ns[l][2]
                    mag = sign(model.layers[l].weight[n2,n1]/M)*maxwidth*(minwidth+abs.(model.layers[l].weight[n2,n1])/M)
                    isnan(mag) ? mag = 0 : nothing
                    push!(newConnectionStrength,mag)
                    push!(newConnectionStrength,mag)
                end
            end
        end
        # propogate colour and magnitude updates
        ConnectionStrength[] = newConnectionStrength
        ConnectionColour[] = maptocolours(newConnectionStrength,cm)

        # display loss in progress
        e = sum([loss(model,data[i][1],data[i][2]) for i in 1:length(data)])/length(data)
        next!(prog,showvalues=[(:iter,k),(:loss,e)])
    end

    return fig
end
