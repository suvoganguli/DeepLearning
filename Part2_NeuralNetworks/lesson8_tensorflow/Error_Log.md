## Error Log for Part2_Neural_Network, Lesson 9 (Tensorflow), Slide 16

#### ----------------------------
####CODE

#### Remove the previous weights and bias
tf.reset_default_graph()

#### Two Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

#### Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Load the weights and bias
    saver.restore(sess, save_file)

    # Show the values of weights and bias
    print('Weight:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))
    
#### ----------------------------
#### Error Log (truncated at the end)

INFO:tensorflow:Restoring parameters from ./answer.ckpt
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
~/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
   1322     try:
-> 1323       return fn(*args)
   1324     except errors.OpError as e:

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
   1301                                    feed_dict, fetch_list, target_list,
-> 1302                                    status, run_metadata)
   1303 

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py in __exit__(self, type_arg, value_arg, traceback_arg)
    472             compat.as_text(c_api.TF_Message(self.status.status)),
--> 473             c_api.TF_GetCode(self.status.status))
    474     # Delete the underlying status object from memory otherwise it stays alive

InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [3] rhs shape= [256,10]
	 [[Node: save/Assign_1 = Assign[T=DT_FLOAT, _class=["loc:@Variable_1"], use_locking=true, validate_shape=true, _device="/job:localhost/replica:0/task:0/device:CPU:0"](Variable_1, save/RestoreV2_1)]]

During handling of the above exception, another exception occurred:

InvalidArgumentError                      Traceback (most recent call last)
<ipython-input-8-8080c1686cdc> in <module>()
     11 with tf.Session() as sess:
     12     # Load the weights and bias
---> 13     saver.restore(sess, save_file)
     14 
     15     # Show the values of weights and bias

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/training/saver.py in restore(self, sess, save_path)
   1664     if context.in_graph_mode():
   1665       sess.run(self.saver_def.restore_op_name,
-> 1666                {self.saver_def.filename_tensor_name: save_path})
   1667     else:
   1668       self._build_eager(save_path, build_save=False, build_restore=True)

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py in run(self, fetches, feed_dict, options, run_metadata)
    887     try:
    888       result = self._run(None, fetches, feed_dict, options_ptr,
--> 889                          run_metadata_ptr)
    890       if run_metadata:
    891         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
   1118     if final_fetches or final_targets or (handle and feed_dict_tensor):
   1119       results = self._do_run(handle, final_targets, final_fetches,
-> 1120                              feed_dict_tensor, options, run_metadata)
   1121     else:
   1122       results = []

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
   1315     if handle is None:
   1316       return self._do_call(_run_fn, self._session, feeds, fetches, targets,
-> 1317                            options, run_metadata)
   1318     else:
   1319       return self._do_call(_prun_fn, self._session, handle, feeds, fetches)

~/anaconda3/lib/python3.6/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
   1334         except KeyError:
   1335           pass
-> 1336       raise type(e)(node_def, op, message)
   1337 
   1338   def _extend_graph(self):

InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [3] rhs shape= [256,10]
	 [[Node: save/Assign_1 = Assign[T=DT_FLOAT, _class=["loc:@Variable_1"], use_locking=true, validate_shape=true, _device="/job:localhost/replica:0/task:0/device:CPU:0"](Variable_1, save/RestoreV2_1)]]
