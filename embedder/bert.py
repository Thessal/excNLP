import os
import sys
import shutil
import tensorflow as tf
import multiprocessing


def initialize(self, files_in, files_out, vocab_file, converted_vocab_file=BERT_VOCAB_FILE, pool_size=7, config={}):
    _monkey_patch_tf_for_bert(tf)
    _prepare_pretraining()
    _run_pretraining(self, bert_config_file, input_files, output_dir)
    _load_bert()

    config["embedder_type"] = "bert"
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

def _tokens_to_tensors(sentences, vocab):
    # sentences = [['[CLS]', *[x[0] + '/' + POS_convert[x[1]] + '_' for x in tokens], '[SEP]'] for tokens in
    #              sentences]
    sentences = [['[CLS]', *[x for x in tokens], '[SEP]'] for tokens in sentences]
    sentences = [x[:min(len(x), self.max_seq_len) - 1] + ['[SEP]'] for x in sentences]
    sentences = [x + ['[PAD]'] * (self.max_seq_len - len(x)) for x in sentences]
    vocab_wrap = defaultdict(lambda: vocab['[UNK]'], vocab)  # FIXME : Fallback to '[UNK]'
    sentence_vectors = [convert_tokens_to_ids(vocab_wrap, tokens) for tokens in sentences]
    return sentence_vectors

def _prepare_pretraining():
    from .bert import create_pretraining_data as prep

    prep.tf.logging.set_verbosity(prep.tf.logging.WARN)  # (prep.tf.logging.ERROR)
    prep.FLAGS.do_lower_case = False
    prep.FLAGS.do_whole_word_mask = False  # Consider setting it True (***** New May 31st, 2019: Whole Word Masking Models *****)
    prep.FLAGS.max_seq_length = 128
    prep.FLAGS.max_predictions_per_seq = 20
    prep.FLAGS.random_seed = 12345
    prep.FLAGS.dupe_factor = 10
    prep.FLAGS.masked_lm_prob = 0.15
    prep.FLAGS.short_seq_prob = 0  # ", 0.1,
    prep.FLAGS.vocab_file = converted_vocab_file

    # Convert vocab into BERT vocab
    # NOTE : It would be nice if we can truncate vocab, but that will cause duplicate keys
    vocab = config["tokenizer_vocab"]["tokens"]
    vocab_special = config["tokenizer_vocab"]["special"]
    bert_special = dict(PADDING="[PAD]", UNKNOWN="[UNK]", CLASSIFICATION="[CLS]", SEPARATOR="[SEP]",
                              MASK="[MASK]", BEGIN="< S >", END="< T >")
    for k,v in bert_special:
        if k in vocab_special.keys():
            vocab[vocab_special[k]["idx"]] = vocab_special[k]["token"]
        else :
            vocab.append(v)
    with open(prep.FLAGS.vocab_file, "w", encoding="utf-8") as tmpf:
        tmpf.write('\n'.join(vocab))

    # Replace tokenizer into my version
    prep.tokenization.FullTokenizer.tokenize = lambda text: tokenizer.tokenize(text, config=config, mark_unk=False)

    # Create pretraining data
    with multiprocessing.Pool(processes=pool_size) as pool:
        results = pool.starmap(self._create_pretraining_data, zip(files_in, files_out))
    print("Errors:")
    print('\n'.join([r for r in results if r]))


def _create_pretraining_data(self, file_in, file_out):
    """
    Convert raw text into BERT train examples
    :param files_in: text file
    :param files_out: BERT train file
    :param vocab_file: sentencepiece vocab file
    :return:
    """
    worker_id = multiprocessing.current_process().name
    file_in_formatted = f"/dev/shm/temp_{worker_id}.txt"
    file_out_tmp = f"/dev/shm/out_{worker_id}.tfrecord"

    # Convert raw text into BERT format text (sentence per line, paragraph separated by empty line)
    doc = Document(file = file_in)
    output = '\n'.join([line['sentence'] for line in doc._generate(unit="sentence", detail=False))])
    with open(file_in_formatted, "w", encoding="utf-8") as tmpf:
        tmpf.write(output)

    # Create pretraining data
    try:
        os.makedirs(os.path.dirname(file_out), exist_ok=True)
        prep.FLAGS.input_file = file_in_formatted
        prep.FLAGS.output_file = file_out_tmp  # file_out : Need to escape comma in filename
        prep.main(None)
        assert (prep.FLAGS.input_file == file_in_formatted)  # ¯\_(ツ)_/¯
        shutil.copy(file_out_tmp, file_out)
    except Exception as e:
        print()
        print("===Error===")
        print(file_in)
        print(e)
        print("===========")
        return file_in + '\t' + str(e)
    return None


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


def _monkey_patch_tf_for_bert(tf):
    # Monkey patch : https://docs.google.com/spreadsheets/d/1FLFJLzg7WNP6JHODX5q8BDgptKafq_slHpnHVbJIteQ/edit#gid=0
    # Monkey patch for bert create_pretraining_data
    tf.flags = tf.compat.v1.flags
    tf.python_io = tf.compat.v1.python_io
    tf.logging = tf.compat.v1.logging
    tf.gfile = tf.compat.v1.gfile
    tf.FixedLenFeature = tf.io.FixedLenFeature
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/embedder/bert")
    tf.train.Optimizer = tf.compat.v1.train.Optimizer

    ## Monkey patch for bert run_pretraining
    tf.parse_single_example = tf.io.parse_single_example
    tf.to_int32 = tf.compat.v1.to_int32
    tf.variable_scope = tf.compat.v1.variable_scope
    tf.get_variable = tf.compat.v1.get_variable
    tf.truncated_normal_initializer = tf.compat.v1.truncated_normal_initializer
    tf.assert_less_equal = tf.compat.v1.assert_less_equal
    tf.layers = tf.compat.v1.layers
    tf.trainable_variables = tf.compat.v1.trainable_variables
    tf.train.get_or_create_global_step = tf.compat.v1.train.get_or_create_global_step
    tf.train.polynomial_decay = tf.compat.v1.train.polynomial_decay
    tf.metrics = tf.compat.v1.metrics

    class Contrib(object):
        tpu = tf.compat.v1.estimator.tpu
        data = tf.data.experimental

        class layers:
            def layer_norm(inputs, begin_norm_axis=-1, begin_params_axis=-1, scope=None):
                """Run layer normalization on the last dimension of the tensor."""
                return tf.keras.layers.LayerNormalization(name=scope, axis=begin_norm_axis, epsilon=1e-12,
                                                          dtype=tf.float32)(inputs)
                # return tf.contrib.layers.layer_norm(
                #     inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

    tf.contrib = Contrib()


def _load_bert(self):
    # https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
    model_ckpt = os.path.join(self.model_dir, self.model_file)
    bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
    # max_seq_len = bert_params.max_position_embeddings

    l_bert = bert.BertModelLayer.from_params(bert_params)
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
    if self.USE_TPU:
        model = tpu_model(inputs=l_input_ids, outputs=output)
        tpu_model.batch_size = self.batch_size
    else:
        model = keras.Model(inputs=l_input_ids, outputs=output)
    model.build(input_shape=(None, self.max_seq_len))

    # (2 segments) provide a custom token_type/segment id as a layer input
    # output = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
    # model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
    # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])

    l_bert.apply_adapter_freeze()
    bert.load_stock_weights(l_bert, model_ckpt)

    # load the fine tuned classifier model weights
    # fine_path = os.path.join(self.model_dir, self.model_file)
    # finetuned_weights = {w.name: w for w in model.trainable_weights}
    # print(finetuned_weights)
    # checkpoint = tf.train.Checkpoint(**finetuned_weights)
    # load_status = checkpoint.restore(fine_path)
    # load_status.assert_consumed().run_restore_ops()

    return model
