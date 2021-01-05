import os
import sys
import shutil
from tokenizer import *
from constant import *
import tensorflow as tf
import multiprocessing
import logging
from absl import flags # NOTE: flags is global variable... so bert scripts are exclusive each other and stateful.

# Monkey patch : https://docs.google.com/spreadsheets/d/1FLFJLzg7WNP6JHODX5q8BDgptKafq_slHpnHVbJIteQ/edit#gid=0
# Monkey patch for bert create_pretraining_data
tf.flags = tf.compat.v1.flags
tf.python_io = tf.compat.v1.python_io
tf.logging = tf.compat.v1.logging
tf.gfile = tf.compat.v1.gfile
tf.FixedLenFeature = tf.io.FixedLenFeature
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/encoder/bert")
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
        def layer_norm(inputs, begin_norm_axis = -1, begin_params_axis=-1, scope=None):
            """Run layer normalization on the last dimension of the tensor."""
            return tf.keras.layers.LayerNormalization(name=scope, axis=begin_norm_axis, epsilon=1e-12, dtype=tf.float32)(inputs)
            # return tf.contrib.layers.layer_norm(
            #     inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

tf.contrib = Contrib()

from .bert import create_pretraining_data as prep




def __init__(self):
    self.shape = (1, 128)
    self.tokenizer = TokenizerSpm(
        TOKENIZER_DIR
    )
    self.tokenizer.load(enable_tf=False)
    return

def bert_preprocess_model(self, text):
    N = self.shape[1]
    output = [y for x in self.tokenizer.tokenize(text) for y in x]
    output = output[:min(len(output), N)]
    return {
        'input_mask': (output + [200000] * (N - len(output))),
        'input_word_ids': [1] * len(output) + [0] * (N - len(output)),
        'input_type_ids': [0] * len(output),
    }

def run_pretraining(self, bert_config_file, input_files, output_dir):
    for name in list(flags.FLAGS):
        delattr(flags.FLAGS, name)
    from .bert import run_pretraining as train
    train.flags = flags

    logger = tf.get_logger()
    logger.setLevel(logging.WARN) #(logging.INFO)
    train.FLAGS.bert_config_file = bert_config_file
    train.FLAGS.input_file = ''  # ','.join(input_files)
    train.FLAGS.output_dir = output_dir
    train.FLAGS.max_seq_length = 128
    train.FLAGS.max_predictions_per_seq = 20
    train.FLAGS.do_train = True
    train.FLAGS.do_eval = True
    if I_AM_POOR:
        train.FLAGS.train_batch_size = 4 #32
        train.FLAGS.eval_batch_size = 1 #8

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

    # # If TPU is not available, this will fall back to normal Estimator on CPU
    # # or GPU.
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

def create_pretraining_data(self, files_in, files_out, vocab_file, converted_vocab_file=BERT_VOCAB_FILE,
                            pool_size=7):
    # for name in list(flags.FLAGS):
    #     delattr(flags.FLAGS, name)

    prep.tf.logging.set_verbosity(prep.tf.logging.WARN) #(prep.tf.logging.ERROR)
    prep.FLAGS.do_lower_case = False
    prep.FLAGS.do_whole_word_mask = False  # Consider setting it True (***** New May 31st, 2019: Whole Word Masking Models *****)
    prep.FLAGS.max_seq_length = 128
    prep.FLAGS.max_predictions_per_seq = 20
    prep.FLAGS.random_seed = 12345
    prep.FLAGS.dupe_factor = 10
    prep.FLAGS.masked_lm_prob = 0.15
    prep.FLAGS.short_seq_prob = 0  # ", 0.1,
    prep.FLAGS.vocab_file = converted_vocab_file

    # Convert sentencepiece vocab into BERT vocab
    if not os.path.isfile(prep.FLAGS.vocab_file):
        with open(vocab_file, "r") as fp:
            vocab = [x.strip().split('\t') for x in fp.readlines()]
            if vocab[0][0] == '< unk >':
                vocab[0][0] = '[UNK]'
            if vocab[1][0] == '< s >':
                vocab[1][0] = '< S >'
            if vocab[2][0] == '< / s >':
                vocab[2][0] = '< T >'

            vocab_terms = [x[0] for x in vocab]
            for x in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "< S >", "< T >"]:
                if x not in vocab_terms:
                    vocab.append([x, '0'])
            # NOTE : this will make problem when truncating vocab size.

            vocab_terms = [x[0] for x in vocab]
            with open(prep.FLAGS.vocab_file, "w", encoding="utf-8") as tmpf:
                tmpf.write('\n'.join(vocab_terms))

    # Replace tokenizer into my version
    if prep.tokenization.FullTokenizer.tokenize != self.tokenizer.tokenize:
        prep.tokenization.FullTokenizer.tokenize = self.tokenizer.tokenize

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

    # Convert raw text into BERT format text
    with open(file_in, "r") as fp:
        lines = [x.strip() for x in fp.readlines()]
        output = '\n'.join(textio.heuristic_formatting(lines))
        with open(file_in_formatted, "w", encoding="utf-8") as tmpf:
            tmpf.write(output)

    # Create pretraining data
    try:
        os.makedirs(os.path.dirname(file_out), exist_ok=True)
        prep.FLAGS.input_file = file_in_formatted
        prep.FLAGS.output_file = file_out_tmp #file_out : Need to escape comma in filename
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
