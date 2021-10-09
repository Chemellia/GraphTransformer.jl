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
function message(mha::MultiHeadAttentionLayer, x_i, x_j)
    key = mha.key_layer(x_i)
    query = mha.query_layer(x_j)
    value = mha.value_layer(x_j)
    # edge_feats = mha.edge_projection_layer(e_ij)
    # Reshape - [out_dim, num_heads]
    key = reshape(key, :, mha.num_heads) 
    query = reshape(query, :, mha.num_heads) 
    value = reshape(value, :, mha.num_heads)
    # edge_feats = reshape(edge_feats, :, mha.num_heads)
    # Add edge features 
    scores = sum(key .* query, dims=1) ./ sqrt(mha.out_dim) # shape - [1, num_heads]
    # scores = scores .* edge_feats
    return vcat(scores, value)
end 

function apply_batch_message(mha::MultiHeadAttentionLayer, i, js, X)
    scores = mapreduce(j -> message(mha, _view(X, i), _view(X, j)), hcat, js)
    n = size(scores, 1)
    alphas = Flux.softmax(reshape(view(scores, 1, :), mha.num_heads, :), dims=2)
    values = view(scores, 2:n, :) .* reshape(alphas, 1, :)
    # Reshape values to [out_dim * num_heads, num_nodes_in_neighbourhood]
    values = reshape(values, mha.out_dim * mha.num_heads, :)
    # Put the tensor with heads concatenated through a dense layer 
    node_features = mha.node_output_layer(values)
    # Aggregate the values of all neighbour nodes 
    return dropdims(sum(node_features, dims=ndims(node_features)), dims=ndims(node_features))
end

function aggregate_neighbors(mha::MultiHeadAttentionLayer, adj, E, V)
    n = size(adj, 1)
    # a vertex must always receive a message from itself
    Zygote.ignore() do
        GraphLaplacians.add_self_loop!(adj, n)
    end
    return mapreduce(i -> apply_batch_message(mha, i, adj[i], V), hcat, 1:n)
end

function propagate(mha::MultiHeadAttentionLayer, fg::FeaturedGraph)
    adj = adjacency_list(fg)
    edge_feats, node_feats = fg.ef, fg.nf 
    node_feats = aggregate_neighbors(mha, adj, edge_feats, node_feats)
    # Appl transformation to edge features
    edge_feats = mha.edge_projection_layer(edge_feats)
    return node_feats, edge_feats
end

function (mha::MultiHeadAttentionLayer)(graph::FeaturedGraph)
    # Multi Headed Attention over neighbours 
    return propagate(mha, graph)
end 
