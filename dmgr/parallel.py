import pandas as pd
import os


def build(files, modules, cfg_dataset, config):
    """

    Args:
        files: [(parallel.csv,parallel.pkl.gz), (parallel.csv,parallel.pkl.gz), ...]

    Returns:

    """
    tokenizer = modules.tokenizers[cfg_dataset["tokenizer"]]
    tokenize = lambda text: tokenizer.tokenize(text, config=config)

    for file_in, file_out in files:
        if os.path.isfile(file_out):
            continue
        df = pd.read_csv(file_in)
        df.rename(inplace=True, columns={'orig': 'high_korean', 'paraphrased': 'low_korean'})
        df['high_korean'] = df['high_korean'].apply(tokenize)
        df['low_korean'] = df['low_korean'].apply(tokenize)
        df.to_pickle(file_out, compression="infer")
