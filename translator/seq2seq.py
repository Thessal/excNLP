import pandas as pd


## Similar to seq2seq, but I use token level encoding, rather than character level.

def train(dataset, config):
    _load_data()
    _define_model()
    _train_model()


def _load_data():
    # Encoder type : one hot encoding + RNN.
    # It is because BERT do not support appropriate decoding method for sequence generation.
    # Maybe we can use BERT to improve quality of the result.
    #
    # Encoding level : sentencepiece token level encoding.
    # Original seq2seq uses character level encoding, but I think we can utilize tokenizer for etymological information.
    # So I set sentencepiece token as one hot encoding unit.
    # But, If I have good hardware and dataset, I can try character level encoding.

    # # For Character level encoding
    # from tokenizer.explode import explode, assemble, CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST
    # korean_characters = sorted(list(set(CHOSUNG_LIST + JUNGSUNG_LIST + JONGSUNG_LIST)))
    # input_characters = korean_characters
    # target_characters = korean_characters
    # for idx, row in df.iterrows():
    #     input_text = explode(row[lang_from])
    #     input_text = explode(row[lang_from])
    #     target_text = explode(row[lang_to])

    lang_source = "low_korean"
    lang_target = "high_korean"
    df = pd.read_csv(dataset)
    num_row_df = df.shape[0]
    max_seq_length = 64

    num_encoder_tokens = num_row_df
    num_decoder_tokens = num_row_df
    max_encoder_seq_length = max_seq_length
    max_decoder_seq_length = max_seq_length

    # One hot encoding
    kor_vocab = config["tokenizer"]["vocab"]  # TODO {token: index}
    source_token_index = kor_vocab
    target_token_index = kor_vocab

    df_tokenized = pd.DataFrame()
    df_tokenized.insert(lang_source, df[lang_source].apply(
        lambda text: ([source_token_index['<S>']] +
                      tokenizer.tokenize(text, config=config)['index'] +
                      [source_token_index['<T>']]
                      )[:max_seq_length]
    ))
    df_tokenized.insert(lang_target, df[lang_target].apply(
        lambda text: (tokenizer.tokenize(text, config=config)['index'] +
                      [target_token_index['<T>']]
                      )[:max_seq_length]
    ))

    def one_hot_encode(idx):
        return ([0] * (idx - 1)) + [1] + ([0] * (max_seq_length - idx))

    df_one_hot_encode = df.apply(lambda tokens:
                                 [one_hot_encode(idx) for idx in tokens]
                                 )
    df_one_hot_encode[lang_source] = df_one_hot_encode[lang_source].apply(
        lambda vectors: [one_hot_encode(source_token_index["[PAD]"])] * (max_seq_length - len(vectors))
    )
    df_one_hot_encode[lang_source] = df_one_hot_encode[lang_source].apply(
        lambda vectors: [one_hot_encode(target_token_index["[PAD]"])] * (max_seq_length - len(vectors))
    )


def _define_model():
    # Build the model
    # Model([encoder_inputs, decoder_inputs], decoder_outputs)


def _train_model():
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
    )
    # Save model
    model.save("s2s")
