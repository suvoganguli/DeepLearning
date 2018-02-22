##README.md

I am getting the following error for ###Build the Graph:

ValueError                                Traceback (most recent call last)
<ipython-input-436-a2d9a7091ca5> in <module>()
     10     input_data_shape = tf.shape(input_text)
     11     cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
---> 12     logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)
     13 
     14     # Probabilities for generating words

<ipython-input-433-a9f72f394a68> in build_nn(cell, rnn_size, input_data, vocab_size, embed_dim)
     24 
     25     # Embedded Input
---> 26     embed = get_embed(input_data, vocab_size, embed_dim)
     27 
     28     Outputs, FinalState = build_rnn(cell, embed)

<ipython-input-431-fdc7247b76d7> in get_embed(input_data, vocab_size, embed_dim)
     11     embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))  # test: 27 x 256
     12 
---> 13     embed = tf.nn.embedding_lookup(embedding, input_data) # test: 50 x 5
     14 
     15     if True:

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/ops/embedding_ops.py in embedding_lookup(params, ids, partition_strategy, name, validate_indices, max_norm)
     99       return clip_ops.clip_by_norm(x, max_norm, axes=list(range(1, ndims)))
    100     return x
--> 101   with ops.name_scope(name, "embedding_lookup", params + [ids]) as name:
    102     np = len(params)  # Number of partitions
    103     params = ops.convert_n_to_tensor_or_indexed_slices(params, name="params")

~/anaconda3/lib/python3.6/contextlib.py in __enter__(self)
     79     def __enter__(self):
     80         try:
---> 81             return next(self.gen)
     82         except StopIteration:
     83             raise RuntimeError("generator didn't yield") from None

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in name_scope(name, default_name, values)
   4217   if values is None:
   4218     values = []
-> 4219   g = _get_graph_from_inputs(values)
   4220   with g.as_default(), g.name_scope(n) as scope:
   4221     yield scope

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in _get_graph_from_inputs(op_input_list, graph)
   3966         graph = graph_element.graph
   3967       elif original_graph_element is not None:
-> 3968         _assert_same_graph(original_graph_element, graph_element)
   3969       elif graph_element.graph is not graph:
   3970         raise ValueError(

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in _assert_same_graph(original_item, item)
   3905   if original_item.graph is not item.graph:
   3906     raise ValueError(
-> 3907         "%s must be from the same graph as %s." % (item, original_item))
   3908 
   3909 

ValueError: Tensor("input:0", shape=(?, ?), dtype=float32) must be from the same graph as Tensor("Variable:0", shape=(6779, 100), dtype=float32_ref).
