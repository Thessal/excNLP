import os
import sys
import shutil
import tensorflow as tf
import multiprocessing
import numpy as np
import bert as bert_for_tf2
from tensorflow import keras

def initialize(model_path="data/models/bert", train_datasets=[], config={}):
    model_ckpt = os.path.join(model_path, "model.ckpt")
    pool_size = 7
    _monkey_patch_tf_for_bert(tf)

    _prepare_pretraining()
    _run_pretraining(self, bert_config_file, input_files, output_dir)


    if "embedder" not in config.keys():
        config["embedder"] = {}
    config["embedder"]["bert"]["dictionary"] = dictionary
    config["embedder"]["bert"]["model"] = _load_bert()
    return config

def embed(self, tokens):
    sentence_vectors = _tokens_to_tensors(tokens, vocab)
    sentence_vectors = np.array(self.embed_sentence(sentence_vectors))

    embed_result = []
    limit = len(sentence_vectors)
    for offset in range(int(int(limit) / self.batch_size)):
        sentence_tensors = tf.convert_to_tensor(sentence_vectors[offset:offset + self.batch_size], dtype=tf.int32)
        result = self.model.predict(sentence_tensors)
        assert result.shape == (self.batch_size, self.bert_params.hidden_size)
        embed_result += result.tolist()
        bert.loader.trace(f"Embedding : {len(embed_result)}/{limit}")
    return embed_result




def _run_pretraining(self, bert_config_file, input_files, output_dir):
    for name in list(flags.FLAGS):
        delattr(flags.FLAGS, name)
    from .bert import run_pretraining as train
    import logging
    from absl import flags  # stateful
    train.flags = flags

    logger = tf.get_logger()
    logger.setLevel(logging.WARN)  # (logging.INFO)
    train.FLAGS.bert_config_file = bert_config_file
    train.FLAGS.input_file = ''  # ','.join(input_files)
    train.FLAGS.output_dir = output_dir
    train.FLAGS.max_seq_length = 128
    train.FLAGS.max_predictions_per_seq = 20
    train.FLAGS.do_train = True
    train.FLAGS.do_eval = True
    if I_AM_POOR:
        train.FLAGS.train_batch_size = 4  # 32
        train.FLAGS.eval_batch_size = 1  # 8

    if not train.FLAGS.do_train and not train.FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = train.modeling.BertConfig.from_json_file(train.FLAGS.bert_config_file)

    tf.io.gfile.makedirs(train.FLAGS.output_dir)

    logger.info("*** Input Files ***")
    for input_file in input_files:
        logger.info("  %s" % input_file)

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        # cluster=tpu_cluster_resolver,
        master=train.FLAGS.master,
        model_dir=train.FLAGS.output_dir,
        save_checkpoints_steps=train.FLAGS.save_checkpoints_steps,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=train.FLAGS.iterations_per_loop,
            num_shards=train.FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = train.model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=train.FLAGS.init_checkpoint,
        learning_rate=train.FLAGS.learning_rate,
        num_train_steps=train.FLAGS.num_train_steps,
        num_warmup_steps=train.FLAGS.num_warmup_steps,
        use_tpu=train.FLAGS.use_tpu,
        use_one_hot_embeddings=train.FLAGS.use_tpu)

    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=train.FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train.FLAGS.train_batch_size,
        eval_batch_size=train.FLAGS.eval_batch_size)

    if train.FLAGS.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", train.FLAGS.train_batch_size)
        train_input_fn = train.input_fn_builder(
            input_files=input_files,
            max_seq_length=train.FLAGS.max_seq_length,
            max_predictions_per_seq=train.FLAGS.max_predictions_per_seq,
            is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=train.FLAGS.num_train_steps)

    if train.FLAGS.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Batch size = %d", train.FLAGS.eval_batch_size)

        eval_input_fn = train.input_fn_builder(
            input_files=input_files,
            max_seq_length=train.FLAGS.max_seq_length,
            max_predictions_per_seq=train.FLAGS.max_predictions_per_seq,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=train.FLAGS.max_eval_steps)

        output_eval_file = os.path.join(train.FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))



def _load_bert(config={}):
    """
    Loads bert model using bert-for-tf2

    Args:
        config:

    Returns:
        bert-for-tf2 model
    """

    model_ckpt = os.path.join(self.model_dir, self.model_file)
    bert_params = bert_for_tf2.params_from_pretrained_ckpt(self.model_dir)
    # max_seq_len = bert_params.max_position_embeddings

    l_bert = bert_for_tf2.BertModelLayer.from_params(bert_params)
    l_input_ids = keras.layers.Input(shape=(self.max_seq_len,), dtype='int32')
    # l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')

    # (1 segment) using the default token_type/segment id 0
    bert_output = l_bert(l_input_ids)  # output: [batch_size, max_seq_len, hidden_size]
    # Pooling layer for sentence vector
    if self.pooling == "default":  # First token ([CLS]) "This output is usually not a good summary of the semantic content ..."
        first_token_tensor = tf.squeeze(bert_output[:, 0:1, :], axis=1)
        output = tf.keras.layers.Dense(bert_params.hidden_size,
                                       activation=tf.tanh,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                           stddev=bert_params.initializer_range))(first_token_tensor)
    if self.pooling == "average":
        output = tf.squeeze(
            tf.keras.layers.AveragePooling1D(pool_size=self.max_seq_len, data_format='channels_last')(bert_output),
            axis=1)
    elif self.pooling == "max":
        output = tf.squeeze(
            tf.keras.layers.MaxPool1D(pool_size=self.max_seq_len, data_format='channels_last')(bert_output),
            axis=1)
    # else if pooling == "median" : # remove zeros and do something
    elif self.pooling == "none":
        output = bert_output
    model = keras.Model(inputs=l_input_ids, outputs=output)
    model.build(input_shape=(None, self.max_seq_len))

    l_bert.apply_adapter_freeze()
    bert_for_tf2.load_stock_weights(l_bert, model_ckpt)

    return model
