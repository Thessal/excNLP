import os
import glob
import json
import pandas as pd
import modules


class Document:
    def __init__(self, dataset_info_file, config):
        with open(dataset_info_file) as json_file:
            dataset_config = json.load(json_file)
        assert dataset_config["dmgr"]["dmgr"] == "text"
        self.files = glob.glob(os.path.join(dataset_config["dataset_path"], "*.gz"))
        self.names = {file: os.path.splitext(os.path.basename(file))[0] for file in self.files}
        self.tokenizer = modules.tokenizers[dataset_config["dmgr"]["tokenizer"]]
        self.config = config

    def __iter__(self):
        for file in self.files:
            df = pd.read_pickle(file, compression="infer")
            for idx, new_df in df.groupby("idx_sentence"):
                tokens = new_df["token"].to_list()
                yield self.tokenizer.detokenize(tokens)

    def iter_file(self, unit="token"):
        """

        Args:
            unit:

        Returns:
            dataframe(index=['idx_paragraph', 'idx_sentence', 'idx_token'], columns=['token/sentence/paragraph'])
        """
        for file in self.files:
            assert (unit in ["token", "sentence", "paragraph"])
            df = pd.read_pickle(file, compression="infer")
            if unit == "token":
                df.set_index(['idx_paragraph', 'idx_sentence', 'idx_token'], inplace=True)
            elif unit == "sentence":
                df_new = df.groupby(['idx_paragraph', 'idx_sentence'])['token'].apply(
                    self.tokenizer.detokenize).reset_index()
                df_new["idx_token_begin"] = df.groupby(['idx_sentence'])['idx_token'].min()
                df_new["idx_token_end"] = df.groupby(['idx_sentence'])['idx_token'].max()
                df_new.rename(columns={"token": "sentence"}, inplace=True)
                df = df_new
            elif unit == "paragraph":
                df_new = df.groupby(['idx_paragraph'])['token'].apply(self.tokenizer.detokenize).reset_index()
                df_new["idx_sentence_begin"] = df.groupby(['idx_paragraph'])['idx_sentence'].min()
                df_new["idx_sentence_end"] = df.groupby(['idx_paragraph'])['idx_sentence'].min()
                df_new["idx_token_begin"] = df.groupby(['idx_paragraph'])['idx_token'].min()
                df_new["idx_token_end"] = df.groupby(['idx_paragraph'])['idx_token'].max()
                df_new.rename(columns={"token": "paragraph"}, inplace=True)
                df = df_new
            yield {"type": "document", "name": self.names[file], "data": df}
