# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf

batchSize = 2  # noqa: N816

memMaxStep = 3  # noqa: N816
memDepth = 3  # noqa: N816
queryMaxStep = 4  # noqa: N816
queryDepth = 3  # noqa: N816
am_attn_size: int = 2
cell_hidden_size = 3
aw_attn_size: int = 2
am_context_size: int = memDepth

root_variable_scope = "LstmAttention"

with tf.variable_scope(root_variable_scope):
    query = tf.get_variable(
        "input",
        initializer=tf.constant(
            [
                0.25,
                -1.5,
                1.0,
                0.25,
                -0.5,
                -1.5,
                0.1,
                1.5,
                0.25,
                0.0,
                0.0,
                0.0,
                0.1,
                -0.125,
                0.25,
                -0.5,
                0.25,
                0.1,
                1.0,
                0.5,
                -1.5,
                0.0,
                0.0,
                0.0,
            ],
            shape=[batchSize, queryMaxStep, queryDepth],
        ),
    )

    querySeqLen = tf.Variable(  # noqa: N816
        tf.constant([queryMaxStep - 1, queryMaxStep - 2], tf.int32),
        name="query_seq_len",
    )

    memory = tf.get_variable(
        "memory",
        initializer=tf.constant(
            [
                0.1,
                -0.25,
                1.0,
                1.0,
                -1.0,
                -1.5,
                1.0,
                0.25,
                -0.125,
                0.1,
                -0.25,
                0.5,
                -0.25,
                -1.25,
                0.25,
                -1.0,
                1.5,
                -1.250,
            ],
            shape=[batchSize, memMaxStep, memDepth],
        ),
    )

    memSeqLen = tf.Variable(tf.constant([memMaxStep, memMaxStep - 1], dtype=tf.int32), name="mem_seq_len")  # noqa: N816

    with tf.variable_scope("fwBahdanau"):
        fw_mem_layer_weights = tf.get_variable(
            "memory_layer/kernel",
            initializer=tf.constant([4.0, 2.0, 0.5, -8.0, -2.0, -2.0], shape=[memDepth, am_attn_size]),
        )

    fw_query_layer_weights = tf.get_variable(
        "bidirectional_rnn/fw/attention_wrapper/bahdanau_attention/query_layer/kernel",
        initializer=tf.constant(
            [-0.125, -0.25, 0.1, -0.125, -0.5, 1.5],
            shape=[cell_hidden_size, am_attn_size],
        ),
    )

    fw_aw_attn_weights = tf.get_variable(
        "bidirectional_rnn/fw/attention_wrapper/attention_layer/kernel",
        initializer=tf.constant(
            [1.5, 1.0, 0.1, -0.25, 0.1, 1.0, -0.25, -0.125, -1.5, -1.5, -0.25, 1.5],
            shape=[am_context_size + cell_hidden_size, aw_attn_size],
        ),
    )

    fw_am_attention_v = tf.get_variable(
        "bidirectional_rnn/fw/attention_wrapper/bahdanau_attention/attention_v",
        initializer=tf.constant([-0.25, 0.1], shape=[am_attn_size]),
    )

    fw_lstm_cell_kernel = tf.get_variable(
        "bidirectional_rnn/fw/attention_wrapper/lstm_cell/kernel",
        initializer=tf.constant(
            [
                -1.0,
                -1.5,
                -0.5,
                -1.5,
                0.1,
                -0.5,
                0.5,
                -1.5,
                -0.25,
                1.0,
                -0.125,
                -0.25,
                -1.0,
                -0.5,
                0.25,
                -0.125,
                -0.25,
                -1.0,
                1.5,
                1.0,
                -1.5,
                0.25,
                0.5,
                0.5,
                1.5,
                -0.5,
                -1.0,
                -0.5,
                0.1,
                1.0,
                0.1,
                -0.5,
                -0.125,
                -1.5,
                0.1,
                1.5,
                1.0,
                -0.5,
                -0.5,
                -1.5,
                -0.125,
                -0.125,
                0.25,
                -0.25,
                -0.25,
                0.1,
                -0.5,
                -0.25,
                0.25,
                -0.5,
                0.1,
                -0.5,
                -0.25,
                0.25,
                0.1,
                0.5,
                -1.5,
                -0.125,
                1.5,
                0.5,
                -1.5,
                1.0,
                0.1,
                -0.5,
                -1.5,
                0.5,
                -1.0,
                0.25,
                -0.25,
                1.0,
                0.25,
                0.5,
                -0.125,
                0.1,
                -1.0,
                -1.0,
                0.1,
                1.5,
                -1.5,
                0.1,
                1.5,
                0.5,
                0.25,
                1.0,
                1.0,
                -1.5,
                -0.25,
                0.5,
                -0.25,
                1.0,
                -1.0,
                0.25,
                -0.5,
                0.5,
                -1.5,
                0.5,
            ],
            shape=[aw_attn_size + queryDepth + cell_hidden_size, 4 * cell_hidden_size],
        ),
    )

    fw_lstm_cell_bias = tf.get_variable(
        "bidirectional_rnn/fw/attention_wrapper/lstm_cell/bias",
        initializer=tf.constant(
            [0.25, -0.25, 0.1, 1.0, 1.5, -1.5, 1.5, -1.0, -0.25, 1.0, -0.25, 1.0],
            shape=[4 * cell_hidden_size],
        ),
    )

    with tf.variable_scope("bwBahdanau"):
        fw_mem_layer_weights = tf.get_variable(
            "memory_layer/kernel",
            initializer=tf.constant([4.0, 2.0, 0.5, -8.0, -2.0, -2.0], shape=[memDepth, am_attn_size]),
        )

    bw_query_layer_weights = tf.get_variable(
        "bidirectional_rnn/bw/attention_wrapper/bahdanau_attention/query_layer/kernel",
        initializer=tf.constant(
            [-0.125, -0.25, 0.1, -0.125, -0.5, 1.5],
            shape=[cell_hidden_size, am_attn_size],
        ),
    )

    bw_aw_attn_weights = tf.get_variable(
        "bidirectional_rnn/bw/attention_wrapper/attention_layer/kernel",
        initializer=tf.constant(
            [1.5, 1.0, 0.1, -0.25, 0.1, 1.0, -0.25, -0.125, -1.5, -1.5, -0.25, 1.5],
            shape=[am_context_size + cell_hidden_size, aw_attn_size],
        ),
    )

    bw_am_attention_v = tf.get_variable(
        "bidirectional_rnn/bw/attention_wrapper/bahdanau_attention/attention_v",
        initializer=tf.constant([-0.25, 0.1], shape=[am_attn_size]),
    )

    bw_lstm_cell_kernel = tf.get_variable(
        "bidirectional_rnn/bw/attention_wrapper/lstm_cell/kernel",
        initializer=tf.constant(
            [
                -1.0,
                -1.5,
                -0.5,
                -1.5,
                0.1,
                -0.5,
                0.5,
                -1.5,
                -0.25,
                1.0,
                -0.125,
                -0.25,
                -1.0,
                -0.5,
                0.25,
                -0.125,
                -0.25,
                -1.0,
                1.5,
                1.0,
                -1.5,
                0.25,
                0.5,
                0.5,
                1.5,
                -0.5,
                -1.0,
                -0.5,
                0.1,
                1.0,
                0.1,
                -0.5,
                -0.125,
                -1.5,
                0.1,
                1.5,
                1.0,
                -0.5,
                -0.5,
                -1.5,
                -0.125,
                -0.125,
                0.25,
                -0.25,
                -0.25,
                0.1,
                -0.5,
                -0.25,
                0.25,
                -0.5,
                0.1,
                -0.5,
                -0.25,
                0.25,
                0.1,
                0.5,
                -1.5,
                -0.125,
                1.5,
                0.5,
                -1.5,
                1.0,
                0.1,
                -0.5,
                -1.5,
                0.5,
                -1.0,
                0.25,
                -0.25,
                1.0,
                0.25,
                0.5,
                -0.125,
                0.1,
                -1.0,
                -1.0,
                0.1,
                1.5,
                -1.5,
                0.1,
                1.5,
                0.5,
                0.25,
                1.0,
                1.0,
                -1.5,
                -0.25,
                0.5,
                -0.25,
                1.0,
                -1.0,
                0.25,
                -0.5,
                0.5,
                -1.5,
                0.5,
            ],
            shape=[aw_attn_size + queryDepth + cell_hidden_size, 4 * cell_hidden_size],
        ),
    )

    bw_lstm_cell_bias = tf.get_variable(
        "bidirectional_rnn/bw/attention_wrapper/lstm_cell/bias",
        initializer=tf.constant(
            [0.25, -0.25, 0.1, 1.0, 1.5, -1.5, 1.5, -1.0, -0.25, 1.0, -0.25, 1.0],
            shape=[4 * cell_hidden_size],
        ),
    )

reuse = tf.AUTO_REUSE  # tf.AUTO_REUSE or TRUE
with tf.variable_scope(root_variable_scope, reuse=reuse):
    with tf.variable_scope("fwBahdanau", reuse=reuse):
        fw_am = tf.contrib.seq2seq.BahdanauAttention(am_attn_size, memory, memSeqLen)
    with tf.variable_scope("bwBahdanau", reuse=reuse):
        bw_am = tf.contrib.seq2seq.BahdanauAttention(am_attn_size, memory, memSeqLen)

    fw_cell = tf.contrib.rnn.LSTMCell(num_units=cell_hidden_size, forget_bias=0.0)
    bw_cell = tf.contrib.rnn.LSTMCell(num_units=cell_hidden_size, forget_bias=0.0)

    fw_attn_wrapper = tf.contrib.seq2seq.AttentionWrapper(
        fw_cell, fw_am, attention_layer_size=aw_attn_size, output_attention=False
    )
    bw_attn_wrapper = tf.contrib.seq2seq.AttentionWrapper(
        bw_cell, bw_am, attention_layer_size=aw_attn_size, output_attention=False
    )

    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        fw_attn_wrapper, bw_attn_wrapper, query, querySeqLen, dtype=tf.float32
    )

tensors = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=root_variable_scope)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run([outputs, states])

    tensors.append(outputs[0])
    tensors.append(outputs[1])

    fw_state = states[0]  # output_state_fw
    cell = fw_state.cell_state.c
    attention = fw_state.attention
    alignments = fw_state.alignments
    sess.run(tf.Print(cell, [cell], "====FinalState(fw)", summarize=10000))
    sess.run(tf.Print(alignments, [alignments], "====Final Alignment(fw)", summarize=10000))
    sess.run(tf.Print(attention, [attention], "====Final Attention Context(fw)", summarize=10000))

    bw_state = states[1]  # output_state_bw
    cell = bw_state.cell_state.c
    attention = bw_state.attention
    alignments = bw_state.alignments
    sess.run(tf.Print(cell, [cell], "====FinalState(bw)", summarize=10000))
    sess.run(tf.Print(alignments, [alignments], "====Final Alignment(bw)", summarize=10000))
    sess.run(tf.Print(attention, [attention], "====Final Attention Context(bw)", summarize=10000))

    for t in tensors:
        shape_str = "[" + ",".join(list(map(lambda x: str(x.__int__()), t.get_shape()))) + "]"
        sess.run(
            tf.Print(
                t,
                [tf.reshape(t, [-1])],
                "\t".join([t.name, shape_str, ""]),
                summarize=10000,
            )
        )
