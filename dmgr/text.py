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
    for file_in, file_out in files:
        if os.path.isfile(file_out):
            continue
        with open(file_in, 'r') as fp:
            indexer = _generate(fp.readlines(), formatter, tokenizer, config, unit="token", detail=False)
            indexed = ((x['begin'][0], x['begin'][1], x['begin'][2], x['text']) for x in indexer)
            df = pd.DataFrame(columns=["idx_paragraph", "idx_sentence", "idx_token", "token"], data=indexed)
        df.to_pickle(file_out, compression="infer")

def _generate(text, formatter, tokenizer, config, unit="token", detail=False):
    """
    get next item and position
    :param unit: 'token','tokens','sentence','sentences','paragraph','paragraphs','document'
    :return:
    """
    if unit == "sentence": unit = "tokens"
    if unit == "paragraph": unit = "sentences"
    if unit == "document": unit = "paragraphs"
    if unit == "token": detail = False
    if detail: unit = unit + "_detail"
    idx_p = 0
    idx_s = 0
    idx_t = 0
    idxs_p = []
    idxs_s = []
    idxs_t = []
    for p in formatter.format(text, config=config):  # For each paragraph
        idxs_s.clear()
        idx_tmp_s = (idx_p, idx_s, idx_t)
        for s in p:
            idxs_t.clear()
            idx_tmp_w = (idx_p, idx_s, idx_t)
            for t in tokenizer.tokenize(s, config=config)["text"]:
                if unit == "token": yield {'text': t, 'begin': (idx_p, idx_s, idx_t)}
                if unit == "tokens_detail": idxs_t.append((idx_p, idx_s, idx_t))
                idx_t += 1
            if unit == "tokens": yield {'text': s, 'begin': idx_tmp_w, 'end': (idx_p, idx_s, idx_t)}
            if unit == "tokens_detail": yield {'element': [{'text': x, 'begin': y} for x, y in zip(s, idxs_t)],
                                               'begin': idx_tmp_w,
                                               'end': (idx_p, idx_s, idx_t)}
            if unit == "sentences_detail": idxs_s.append(idx_tmp_w)
            idx_s += 1
        if unit == "sentences": yield {'text': p, 'begin': idx_tmp_s, 'end': (idx_p, idx_s, idx_t)}
        if unit == "sentences_detail": yield {'element': [{'text': x, 'begin': y} for x, y in zip(s, idxs_s)],
                                              'begin': idx_tmp_s,
                                              'end': (idx_p, idx_s, idx_t)}
        if unit == "paragraphs_detail": idxs_p.append(idx_tmp_s)
        idx_p += 1
