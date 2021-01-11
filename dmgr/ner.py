import pandas as pd
import os


def build(files, modules, cfg_dataset, config):
    """

    Args:
        files: [(raw_file.txt, dataset_file.gz), ...]
        formatter:
        tokenizer:

    Returns:

    """
    formatter = modules.formatters[cfg_dataset["formatter"]]
    tokenizer = modules.tokenizers[cfg_dataset["tokenizer"]]
    raise NotImplementedError