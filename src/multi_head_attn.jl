struct MultiHeadAttentionLayer{L,I,B} <: MessagePassing 
    query_layer::L
    key_layer::L
    value_layer::L
    edge_projection_layer::L
    node_output_layer::L  
    in_dim::I 
    out_dim::I 
    num_heads::I 
    use_bias::B 
end 

function MultiHeadAttentionLayer(
    in_dim::Int, 
    out_dim::Int, 
    num_heads::Int; 
    bias=true, 
    init=glorot_uniform
    )
    query_layer = Flux.Dense(in_dim, out_dim * num_heads, bias=bias, init=init)
    key_layer = Flux.Dense(in_dim, out_dim * num_heads, bias=bias, init=init)
    value_layer = Flux.Dense(in_dim, out_dim * num_heads, bias=bias, init=init)
    edge_projection_layer = Flux.Dense(in_dim, out_dim * num_heads, bias=bias, init=init)
    # The final projection layer for the output of attention module 
    node_output_layer = Flux.Dense(out_dim * num_heads, out_dim * num_heads, bias=bias, init=init)
    return MultiHeadAttentionLayer(query_layer, 
                                    key_layer, 
                                    value_layer, 
                                    edge_projection_layer, 
                                    node_output_layer, 
                                    in_dim, out_dim, num_heads, bias)
end 

@functor MultiHeadAttentionLayer

# The message function 
"""
Arguments: 
- `x_i`: Feature vector of i'th node 
- `x_j`: Feature vector of j'th node, where j is in the neighbourhood of i 
- `e_ij`: Feature vector for the edge between i and j 
"""
function message(mha::MultiHeadAttentionLayer, i, j, edge_index_map, X, E)
    x_i = _view(X, i)
    x_j = _view(X, j)

    
    key = mha.key_layer(x_i)
    query = mha.query_layer(x_j)
    value = mha.value_layer(x_j)
    
    if i != j 
        edge_index = edge_index_map[(i, j)]
        edge_feat = _view(E, edge_index)
    else 
        # If it is self edge then just use ones 
        edge_feat = ones(size(value))
    end 

    # Reshape - [out_dim, num_heads]
    key = reshape(key, :, mha.num_heads) 
    query = reshape(query, :, mha.num_heads) 
    value = reshape(value, :, mha.num_heads)
    edge_feat = reshape(edge_feat, :, mha.num_heads)

    # Add edge features 
    scores = sum(key .* query .* edge_feat, dims=1) ./ sqrt(mha.out_dim) # shape - [1, num_heads]
    # scores = scores .* edge_feats
    return vcat(scores, value)
end 

function edge_message(mha::MultiHeadAttentionLayer, i, j, edge_index_map, E)
    if i != j 
        edge_index = edge_index_map[(i, j)]
        edge_feat = _view(E, edge_index)
    else 
        # If it is self edge then just use ones 
        edge_feat = zeros(mha.out_dim * mha.num_heads)
    end 
    return edge_feat
end 

function apply_batch_message(mha::MultiHeadAttentionLayer, i, js, edge_index_map, X, E)
    scores = mapreduce(j -> message(mha, i, j, edge_index_map, X, E), hcat, js)
    nodes = size(scores, 1)
    # score shape - [num_heads, num_nodes_in_neighbourhood]
    reshaped_scores = reshape(view(scores, 1, :), mha.num_heads, :)
    alphas = Flux.softmax(reshaped_scores, dims=2)
    # alpha shape - [1, num_heads, num_nodes_in_neighbourhood]
    alphas = reshape(alphas, 1, mha.num_heads, :)

    values = view(scores, 2:nodes, :) 
    values = reshape(values, mha.out_dim, mha.num_heads, :) .* alphas 
    # Reshape values to [out_dim * num_heads, num_nodes_in_neighbourhood]
    values = reshape(values, mha.out_dim * mha.num_heads, :)
    # Put the tensor with heads concatenated through a dense layer 
    node_features = mha.node_output_layer(values)
    # Aggregate the values of all neighbour nodes
    node_features = dropdims(sum(node_features, dims=ndims(node_features)), dims=ndims(node_features))

    # Edge message passing 
    edge_feats = mapreduce(j -> edge_message(mha, i, j, edge_index_map, E), hcat, js)
    edge_feats = reshape(edge_feats, mha.out_dim, mha.num_heads, :)
    edge_feats = edge_feats .* reshape(reshaped_scores, 1, mha.num_heads, :)
    edge_feats = reshape(edge_feats, mha.out_dim * mha.num_heads, :)
    # Edge projection layer 
    edge_features = mha.edge_projection_layer(edge_feats)
    # edge_features = dropdims(sum(edge_feats, dims=ndims(edge_feats)), dims=ndims(edge_feats))
    
    return node_features, edge_features
end

function aggregate_neighbors(mha::MultiHeadAttentionLayer, adj, edge_index_map, E, V)
    n = size(adj, 1)
    # # a vertex must always receive a message from itself
    # Zygote.ignore() do
    #     GraphLaplacians.add_self_loop!(adj, n)
    # end
    multi_hcat = (args...) -> hcat.(args...)
    return mapreduce(i -> apply_batch_message(mha, i, adj[i], edge_index_map, V, E), multi_hcat, 1:n)
end

function propagate(mha::MultiHeadAttentionLayer, fg::FeaturedGraph)
    adj = adjacency_list(fg)
    
    # Get the edge map to retrieve edges 
    edge_index_map = edge_index_table(adj, fg.directed) 
    edge_feats, node_feats = fg.ef, fg.nf 

    # Attention propagation 
    node_feats, edge_feats = aggregate_neighbors(mha, adj, edge_index_map, edge_feats, node_feats)

    return node_feats, edge_feats
end

function (mha::MultiHeadAttentionLayer)(graph::FeaturedGraph)
    # Multi Headed Attention over neighbours 
    return propagate(mha, graph)
end 
