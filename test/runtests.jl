using Test
using Flux 
using Flux:params
using GraphTransformer
using GeometricFlux
using LightGraphs

@testset "GraphTransformer.jl" begin
    include("./multi_head_attn.jl")
    include("./graph_transformer_layer.jl")
end
