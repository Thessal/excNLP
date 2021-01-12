import tensorflow as tf
import os
from dmgr.builder import read_json
import pandas as pd

def initialize(model_path, train_dataset='NER',config={}):
    input_dataset_config = read_json(os.path.join("data/datasets/",train_dataset+'.json'))
    dataset_file = os.path.join(input_dataset_config['dataset_path'],'processed.pkl.gz')
    data = pd.read_pkl(dataset_file,compression="infer")

    bert_model = config["embedder"]["bert"]["model"]
    bert_model_pooled = config["embedder"]["bert"]["pooled_model"]

    # dump to model_path

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
