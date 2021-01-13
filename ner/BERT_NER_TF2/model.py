from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

from .bert_modeling import BertConfig, BertModel
from .utils import tf_utils


class BertNer(tf.keras.Model):

    def __init__(self, ckpt_dir, float_type, num_labels, max_seq_length, final_layer_initializer=None):
        '''
        ckpt_dir : string. checkpoint path.
        float_type : tf.float32
        num_labels : num of tags in NER task
        max_seq_length : max_seq_length of tokens
        final_layer_initializer : default:  tf.keras.initializers.TruncatedNormal
        '''
        super(BertNer, self).__init__()

        max_seq_len = 128
        l_input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32')

        # Origianal config:
        # input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        # input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        # input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
        # bert_layer = BertModel(config=bert_config, float_type=float_type)
        # bert_output = bert_layer(l_input_ids)
        # _, sequence_output = bert_layer(input_word_ids, input_mask, input_type_ids)
        # self.bert = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=[sequence_output])

        import bert as bert_for_tf2
        bert_params = bert_for_tf2.params_from_pretrained_ckpt(ckpt_dir)
        l_bert = bert_for_tf2.BertModelLayer.from_params(bert_params)
        bert_output = l_bert(l_input_ids)  # output: [batch_size, max_seq_len, hidden_size]
        # TODO : use self.bert = config["embedder"]["bert"]["model"],
        #  rather than calling params_from_pretrained_ckpt again.

        self.bert = tf.keras.Model(inputs=l_input_ids, outputs=bert_output)
        self.dropout = tf.keras.layers.Dropout(
            rate=bert_params["hidden_dropout"])

        if final_layer_initializer is not None:
            initializer = final_layer_initializer
        else:
            initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_params["initializer_range"])

        self.classifier = tf.keras.layers.Dense(
            num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=float_type)

    def call(self, input_word_ids, input_mask=None, input_type_ids=None, valid_ids=None, **kwargs):
        #sequence_output = self.bert([input_word_ids, input_mask, input_type_ids], **kwargs) # TODO : add mask to my bert model
        sequence_output = self.bert(input_word_ids, **kwargs)

        valid_output = []
        for i in range(sequence_output.shape[0]):
            r = 0
            temp = []
            for j in range(sequence_output.shape[1]):
                if valid_ids[i][j] == 1:
                    temp = temp + [sequence_output[i][j]]
                else:
                    r += 1
            temp = temp + r * [tf.zeros_like(sequence_output[i][j])]
            valid_output = valid_output + temp
        valid_output = tf.reshape(tf.stack(valid_output), sequence_output.shape)
        sequence_output = self.dropout(
            valid_output, training=kwargs.get('training', False))
        logits = self.classifier(sequence_output)
        return logits
