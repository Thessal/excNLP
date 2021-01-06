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
        self.files = glob.glob(os.path.join(dataset_config["dataset_path"],"*.gz"))
        self.tokenizer = modules.tokenizers[dataset_config["dmgr"]["tokenizer"]]
        self.config = config

    def __iter__(self):
        for file in self.files :
            df = pd.read_pickle(file,compression="infer")
            for idx, new_df in df.groupby("idx_sentence"):
                tokens = new_df["token"].to_list()
                yield self.tokenizer.detokenize({'text': tokens})
