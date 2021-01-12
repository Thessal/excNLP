import tensorflow as tf
import os
from dmgr.builder import read_json
import pandas as pd
import numpy as np
# from fastprogress import master_bar, progress_bar
import math
from .BERT_NER_TF2.model import BertNer
from .BERT_NER_TF2.optimization import AdamWeightDecay, WarmUp


def _pad(x, size, pad):
    return x[:size] + [pad] * max(0, size - len(x))

def _setup_train(bert_model, learning_rate, train_batch_size, num_train_epochs, warmup_proportion,
                 weight_decay, adam_epsilon, max_seq_length, num_labels, len_train_features):
    num_train_optimization_steps = int(
        len_train_features/ train_batch_size) * num_train_epochs
    warmup_steps = int(warmup_proportion *
                       num_train_optimization_steps)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate,
                                                                     decay_steps=num_train_optimization_steps,
                                                                     end_learning_rate=0.0)
    if warmup_steps:
        learning_rate_fn = WarmUp(initial_learning_rate=learning_rate,
                                  decay_schedule_fn=learning_rate_fn,
                                  warmup_steps=warmup_steps)
    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=adam_epsilon,
        exclude_from_weight_decay=['layer_norm', 'bias'])

    ner = BertNer(bert_model, tf.float32, num_labels, max_seq_length)
    # loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss11_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                               reduction=tf.keras.losses.Reduction.NONE)
    l_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def initialize(model_path, train_dataset='NER', config={},
               train_batch_size=32, num_train_epochs=3, warmup_proportion=0.1,
               learning_rate=5e-5, weight_decay=0.01, adam_epsilon=1e-8):
    input_dataset_config = read_json(os.path.join("data/datasets/", train_dataset + '.json'))
    vocab_file = os.path.join(f'data/datasets/TEXT_BERT/bert.vocab') # TODO : dmgr NER need to dump 'data/datasets/NER/bert.vocab'
    dataset_file = os.path.join(input_dataset_config['dataset_path'], 'processed.pkl.gz')

    bert_model = config["embedder"]["bert"]["model"]
    bert_model_pooled = config["embedder"]["bert"]["pooled_model"]
    input_size = config["embedder"]["bert"]["max_seq_len"]

    preparation = _prepare_data(dataset_file, vocab_file, input_size)
    shuffled_train_data = preparation['shuffled_train_data']
    len_train_features = preparation['len_train_features']
    len_label_index = preparation['len_label_index']
    batched_train_data = shuffled_train_data.batch(train_batch_size)

    ner = _setup_train(bert_model, learning_rate, train_batch_size, num_train_epochs,
                       warmup_proportion, weight_decay, adam_epsilon, max_seq_length=input_size,
                       num_labels = len_label_index, len_train_features= len_train_features)

    loss_metric = tf.keras.metrics.Mean()
    pb_max_len = math.ceil(float(len_train_features) / float(train_batch_size))

    # def train_step(input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask):
    #     with tf.GradientTape() as tape:
    #         logits = ner(input_ids, input_mask, segment_ids, valid_ids,
    #                      training=True)  # batchsize, max_seq_length, num_labels
    #         label_ids_masked = tf.boolean_mask(label_ids, label_mask)
    #         logits_masked = tf.boolean_mask(logits, label_mask)
    #         loss = loss_fct(label_ids_masked, logits_masked)
    #
    #     grads = tape.gradient(loss, ner.trainable_variables)
    #     optimizer.apply_gradients(list(zip(grads, ner.trainable_variables)))
    #     return loss
    #
    # for epoch in epoch_bar:
    #     for (input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask) in progress_bar(batched_train_data, total=pb_max_len, parent=epoch_bar):
    #         loss = train_step(input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask)
    #         loss_metric(loss)
    #         epoch_bar.child.comment = f'loss : {loss_metric.result()}'
    #     loss_metric.reset_states()
    #
    # # model weight save
    # ner.save_weights(os.path.join(args.output_dir, "model.h5"))
    # # copy vocab to output_dir
    # shutil.copyfile(os.path.join(args.bert_model, "vocab.txt"), os.path.join(args.output_dir, "vocab.txt"))
    # # copy bert config to output_dir
    # shutil.copyfile(os.path.join(args.bert_model, "bert_config.json"),
    #                 os.path.join(args.output_dir, "bert_config.json"))
    # # save label_map and max_seq_length of trained model
    # model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
    #                 "max_seq_length": args.max_seq_length, "num_labels": num_labels,
    #                 "label_map": label_map}
    # json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"), indent=4)
    return config

    # model weight save
    ner.save_weights(os.path.join(args.output_dir, "model.h5"))
    # copy vocab to output_dir
    shutil.copyfile(os.path.join(args.bert_model, "vocab.txt"), os.path.join(args.output_dir, "vocab.txt"))
    # copy bert config to output_dir
    shutil.copyfile(os.path.join(args.bert_model, "bert_config.json"), os.path.join(args.output_dir, "bert_config.json"))
    # save label_map and max_seq_length of trained model
    model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                    "max_seq_length": args.max_seq_length, "num_labels": num_labels,
                    "label_map": label_map}
    json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"), indent=4)


def _prepare_data(dataset_file,vocab_file,input_size):
    df = pd.read_pickle(dataset_file, compression="infer")
    with open(vocab_file, 'r') as fp:
        vocab = [x.strip() for x in fp.readlines()]
    print(vocab)
    special_index = {
        '[UNK]': vocab.index('[UNK]'),
        '[PAD]': vocab.index('[PAD]'),
        '< S >': vocab.index('< S >'),
        '< T >': vocab.index('< T >'),
        '[CLS]': vocab.index('[CLS]'),
        '[SEP]': vocab.index('[SEP]'),
    }
    label_index = {'O': 1, 'MISC': 2, 'NUM': 3, 'TIM': 4, 'ORG': 5, 'PER': 6, 'LOC': 7,
                   '[ CLS ]': 8, '[ SEP ]': 9}

    train_features = []
    for index, row in df.iterrows():
        tokens = [special_index['[CLS]']] + row["index"] + [special_index['[SEP]']]
        tags = [label_index[x] for x in (['[CLS]'] + row["tags"] + ['[SEP]'])]

        train_feature = {
            'input_ids': _pad(tokens, input_size, special_index["[PAD]"]),
            'input_mask': _pad([1] * len(tokens), input_size, special_index["[PAD]"]),
            'segment_ids': [0] * input_size,
            'label_ids': _pad(tags, input_size, 0),
            'label_mask': _pad([True] * len(tags), input_size, False),
            'valid_ids': _pad([(1 if token.startswith('_') else 0) for token in tokens], input_size, 0),
        }
        train_features.append(train_feature)

    # https://github.com/bhuvanakundumani/BERT-NER-TF2/blob/4a6bdbb873e344e515f72d88e8acdb6c3a2b3acb/run_ner.py
    all_input_ids = tf.data.Dataset.from_tensor_slices(np.asarray([f["input_ids"] for f in train_features]))
    all_input_mask = tf.data.Dataset.from_tensor_slices(np.asarray([f["input_mask"] for f in train_features]))
    all_segment_ids = tf.data.Dataset.from_tensor_slices(np.asarray([f["segment_ids"] for f in train_features]))
    all_valid_ids = tf.data.Dataset.from_tensor_slices(np.asarray([f["valid_ids"] for f in train_features]))
    all_label_mask = tf.data.Dataset.from_tensor_slices(np.asarray([f["label_mask"] for f in train_features]))
    all_label_ids = tf.data.Dataset.from_tensor_slices(np.asarray([f["label_id"] for f in train_features]))

    # Dataset using tf.data
    train_data = tf.data.Dataset.zip(
        (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids, all_label_mask))
    shuffled_train_data = train_data.shuffle(buffer_size=int(len(train_features) * 0.1),
                                             seed=12345,
                                             reshuffle_each_iteration=True)
    return {"shuffled_train_data" : shuffled_train_data ,
            "len_train_features" : len(train_features),
            "len_label_index" : len(label_index)}
# hub_classifier, hub_encoder = bert.bert_models.classifier_model(
#     # Caution: Most of `bert_config` is ignored if you pass a hub url.
#     bert_config=bert_config, hub_module_url=hub_url_bert, num_labels=2)
#
#
# bert_classifier, bert_encoder = bert.bert_models.classifier_model(
#     bert_config, num_labels=2)
#
#
#
# metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
# bert_classifier.compile(
#     optimizer=optimizer,
#     loss=loss,
#     metrics=metrics)
#
# bert_classifier.fit(
#       glue_train, glue_train_labels,
#       validation_data=(glue_validation, glue_validation_labels),
#       batch_size=32,
#       epochs=epochs)
#
#
# result = bert_classifier(my_examples, training=False)
#
# result = tf.argmax(result).numpy()
# result
#
# reloaded = tf.saved_model.load(export_dir)
