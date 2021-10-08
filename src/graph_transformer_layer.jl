# FeedForward Networks used after self-attention 
"""
    FeedForwardNetwork
"""
struct FFN{L,F,D}
    ffn_layer1::L 
    ffn_layer2::L 
    activation::F 
    dropout::D
end 

function FFN(in_dim::Integer, 
    hidden_dim::Integer; 
    dropout=0.0, 
    activation=Flux.relu, 
    bias=true, 
    init=glorot_uniform)
    ffn_layer1 = Flux.Dense(in_dim, hidden_dim, bias=bias, init=init)
    ffn_layer2 = Flux.Dense(hidden_dim, in_dim, bias=bias, init=init)
    dropout_layer = Flux.Dropout(dropout)
    return FFN(ffn_layer1, ffn_layer2, activation, dropout_layer)
end 

@functor FFN

function (ffn::FFN)(x)
    output = ffn.activation.(ffn.ffn_layer1(x)) 
    output = ffn.dropout(output)
    return ffn.ffn_layer2(output)
end 

"""
    GraphTransformerLayer 
"""
struct GraphTransformerLayer{M,L,F,N}
    attention_layer::M 
    node_output_layer::L 
    edge_output_layer::L 
    node_ffn_layer::F 
    edge_ffn_layer::F 
    node_normalization1::N 
    edge_normalization1::N
    node_normalization2::N 
    edge_normalization2::N  
end 

function GraphTransformerLayer(in_dim::Integer, 
                                out_dim::Integer, 
                                num_heads::Integer; 
                                dropout=0.0, 
                                bias=true, 
                                init=glorot_uniform)
    attention_layer = MultiHeadAttentionLayer(in_dim, out_dim, num_heads, bias=bias, init=init)
    node_output_layer = Flux.Dense(out_dim, out_dim) 
    edge_output_layer = Flux.Dense(out_dim, out_dim) 

    # FFN layer for nodes and edges 
    node_ffn_layer = FFN(out_dim, out_dim * 2, dropout=dropout, bias=bias, init=init)
    edge_ffn_layer = FFN(out_dim, out_dim * 2, dropout=dropout, bias=bias, init=init)

    # Normalization layers 
    node_normalization1 = Flux.LayerNorm(out_dim)
    edge_normalization1 = Flux.LayerNorm(out_dim)
    node_normalization2 = Flux.LayerNorm(out_dim)
    edge_normalization2 = Flux.LayerNorm(out_dim)

    return GraphTransformerLayer(
                attention_layer, 
                node_output_layer, edge_output_layer, 
                node_ffn_layer, edge_ffn_layer, 
                node_normalization1, edge_normalization1, 
                node_normalization2, edge_normalization2)
end 

@functor GraphTransformerLayer 

function (GTL::GraphTransformerLayer)(graph::FeaturedGraph)
    # Propagate Multi-Head Attention 
    graph = GTL.attention_layer(graph)

    node_features = GTL.node_output_layer(graph.nf)
    edge_features = GTL.edge_output_layer(graph.ef)

    # TODO: Fix the first skip connection
    # # Skip connections 
    # node_features += graph.nf
    # edge_features += graph.ef

    # Normalization 
    normalized_node_features = GTL.node_normalization1(node_features)
    normalized_edge_features = GTL.edge_normalization1(edge_features)

    # FFN transformation 
    node_features = GTL.node_ffn_layer(normalized_node_features)
    edge_features = GTL.edge_ffn_layer(normalized_edge_features)

    # Final skip connections 
    node_features += normalized_node_features 
    edge_features += normalized_edge_features

    # Final normalization 
    node_features = GTL.node_normalization2(node_features)
    edge_features = GTL.edge_normalization2(edge_features)

    # Final graph 
    return FeaturedGraph(graph, nf=node_features, ef=edge_features)
end 
