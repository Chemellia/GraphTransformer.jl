@testset "Graph Transformer Layer" begin
    num_nodes = 4 
    num_edges = 10
    hidden_dim = 16
    num_heads = 2 
    g = SimpleDiGraph(num_nodes)
    add_edge!(g, 1, 1)
    add_edge!(g, 2, 2)
    add_edge!(g, 3, 3)
    add_edge!(g, 4, 4)
    add_edge!(g, 1, 2)
    add_edge!(g, 1, 3) 
    add_edge!(g, 1, 4)
    add_edge!(g, 2, 3) 
    add_edge!(g, 2, 4)
    add_edge!(g, 3, 4)
    node_features = rand(hidden_dim, num_nodes)    # [feature_dim, num_num_nodes]
    edge_features = rand(hidden_dim, num_edges)   # [feature_dim, num_num_edges]
    graph = FeaturedGraph(g, nf=node_features, ef=edge_features, directed=:directed)

    layer = GraphTransformerLayer(hidden_dim, 2)
    output_graph = layer(graph)
    @test size(output_graph.nf) == (hidden_dim, num_nodes) 
    @test size(output_graph.ef) == (hidden_dim, num_edges)

    # Test the gradients 
    y = rand(hidden_dim, num_nodes + num_edges)
    grads = gradient(params(layer)) do
        new_graph = layer(graph)
        logits = cat(new_graph.nf, new_graph.ef, dims=2)
        return Flux.logitcrossentropy(logits, y)
    end
    for param in params(layer)
        @test size(param) == size(grads[param])
    end
end
