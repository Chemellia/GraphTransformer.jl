@testset "Graph Transformer Layer" begin
    nodes = 10  
    edges = 5
    hidden_dim = 16 
    num_heads = 2 
    graph = SimpleGraph(nodes)
    add_edge!(graph, 1, 2)
    add_edge!(graph, 1, 3) 
    add_edge!(graph, 1, 4)
    add_edge!(graph, 2, 3) 
    add_edge!(graph, 3, 4)
    node_features = rand(hidden_dim, nodes);    # [feature_dim, num_nodes]
    edge_features = rand(hidden_dim, edges)     # [feature_dim, num_edges]
    graph = FeaturedGraph(graph, nf=node_features, ef=edge_features)

    layer = GraphTransformerLayer(hidden_dim, 2)
    output_graph = layer(graph)
    @test size(output_graph.nf) == (hidden_dim, nodes) 
    @test size(output_graph.ef) == (hidden_dim, edges)
end
