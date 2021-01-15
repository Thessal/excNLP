# Bert train result
tf.keras.utils.plot_model(config["embedder"]["bert"]["model"], show_shapes=True, dpi=48)

# NER fine tuning info
hist = config["ner"]["bert_ner"]["train_loss_history"]
print({k:float(f"{v:.3f}") for k,v in hist.items()})
