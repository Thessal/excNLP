import os
import sys
import glob
import shutil
import multiprocessing
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "dmgr/bert")


def build(files, modules, cfg_dataset, config):
    """
    Builds BERT dataset. This dataset is derived from text datasets.
    """
    if all([os.path.isfile(file_out) for file_in, file_out in files]):
        print("Dataset is recent. Skipping build.")
        return

    _monkey_patch_tf_for_bert(tf)

    vocab = config["tokenizer"][cfg_dataset["tokenizer"]]["vocab"]
    bert_vocab_path = os.path.join(os.path.dirname(files[0][1]),"bert.vocab")
    _convert_vocab_file(vocab, bert_vocab_path)

    formatter = modules.formatters[cfg_dataset["formatter"]]
    tokenizer = modules.tokenizers[cfg_dataset["tokenizer"]]
    files = [(file_in, os.path.splitext(file_out)[0] + ".tfrecord") for file_in, file_out in files]
    _prepare_pretraining(files, vocab_file=bert_vocab_path, tokenizer=tokenizer, formatter=formatter,
                         config=config, unknown_token=vocab["special"]["UNKNOWN"]["token"])


def _convert_vocab_file(vocab, bert_vocab_path):
    vocab_special = vocab["special"]
    tokens = vocab["tokens"]
    bert_special = dict(PADDING="[PAD]", UNKNOWN="[UNK]", CLASSIFICATION="[CLS]", SEPARATOR="[SEP]",
                        MASK="[MASK]", BEGIN="< S >", END="< T >")
    for k in bert_special:
        if k in vocab_special.keys():
            tokens[vocab_special[k]["idx"]] = bert_special[k]
        else:
            tokens.append(bert_special[k])
    with open(bert_vocab_path, "w", encoding="utf-8") as tmpf:
        tmpf.write('\n'.join(tokens))


def _prepare_pretraining(files, vocab_file, tokenizer, formatter, config, unknown_token):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "dmgr/bert"))
    global prep
    from .bert import create_pretraining_data as prep

    prep.tf.logging.set_verbosity(prep.tf.logging.WARN)
    prep.FLAGS.do_lower_case = False
    prep.FLAGS.do_whole_word_mask = False  # Consider setting it True
    prep.FLAGS.max_seq_length = 128
    prep.FLAGS.max_predictions_per_seq = 20
    prep.FLAGS.random_seed = 12345
    prep.FLAGS.dupe_factor = 10
    prep.FLAGS.masked_lm_prob = 0.15
    prep.FLAGS.short_seq_prob = 0  # ", 0.1,
    prep.FLAGS.vocab_file = vocab_file

    # Replace tokenizer into my version
    prep.tokenization.FullTokenizer.tokenize = lambda self,text: \
        [('[UNK]' if x==unknown_token else x ) # sentencepiece specific
         for x in tokenizer.tokenize(text, config=config, mark_unk=True)["text"]]

    # Remove files if already processed
    files = [(file_in, file_out) for file_in, file_out in files if not os.path.isfile(file_out)]

    # Create pretraining data
    assert (os.path.isdir("/dev/shm"))
    prep.tokenization._prepare_pretraining_formatter = lambda x : formatter.format(x, config=config) # ¯\_(ツ)_/¯
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(_create_pretraining_data,
                               files
                               )
    print("Errors:")
    print('\n'.join([r for r in results if r]))


def _create_pretraining_data(file_in, file_out):
    """
    Convert raw text into BERT train examples
    :param file: (file_in, file_out)
    :param vocab_file: sentencepiece vocab file
    :return:
    """
    worker_id = multiprocessing.current_process().name
    #file_in, file_out = file
    file_in_formatted = f"/dev/shm/temp_{worker_id}.txt"
    file_out_tmp = f"/dev/shm/out_{worker_id}.tfrecord"

    # Convert raw text into BERT format text
    # Bert format :
    # one sentence per line.
    # Documents are delimited by empty lines.
    with open(file_in, "r") as fp:
        raw = fp.readlines()
    formatted = prep.tokenization._prepare_pretraining_formatter(raw) # ignoring config for multiprocessing
    paragraphs = ['\n'.join([p for p in paragraph if p.strip()]) for paragraph in formatted]
    bert_text = '\n\n'.join(paragraphs)  # TODO : make test for this
    with open(file_in_formatted, "w") as tmpf:
        tmpf.write(bert_text)

    # Create pretraining data
    try:
        os.makedirs(os.path.dirname(file_out), exist_ok=True)
        prep.FLAGS.input_file = file_in_formatted
        prep.FLAGS.output_file = file_out_tmp
        prep.main(None)
        assert (prep.FLAGS.input_file == file_in_formatted)
        shutil.copy(file_out_tmp, file_out)
    except Exception as e:
        print()
        print("===Error===")
        print(file_in)
        print(e)
        print("===========")
        return file_in + '\t' + str(e)
    return None


def _monkey_patch_tf_for_bert(tf):
    # Monkey patch : https://docs.google.com/spreadsheets/d/1FLFJLzg7WNP6JHODX5q8BDgptKafq_slHpnHVbJIteQ/edit#gid=0
    # Monkey patch for bert create_pretraining_data
    tf.flags = tf.compat.v1.flags
    tf.python_io = tf.compat.v1.python_io
    tf.python_io.TFRecordWriter.__init__old__ = tf.python_io.TFRecordWriter.__init__
    tf.python_io.TFRecordWriter.__init__ = lambda self, path : tf.python_io.TFRecordWriter.__init__old__(self,path,options=tf.compat.v1.io.TFRecordCompressionType.GZIP)
    tf.logging = tf.compat.v1.logging
    tf.gfile = tf.compat.v1.gfile
    tf.FixedLenFeature = tf.io.FixedLenFeature
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
