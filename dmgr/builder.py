import glob
import os
import hashlib
import json
import modules
import dmgr.text
import dmgr.text_bert
import unicodedata

dmgrs = {"text": dmgr.text, "text_bert": dmgr.text_bert}


def build(files, cfg_dataset, config, name="dmgr"):
    print(f'{len(files)} files : {name}')
    dmgrs[cfg_dataset["dmgr"]].build(files, modules, cfg_dataset, config)


def build_all(datasets, config):
    for dataset_name in datasets:
        cfg_dataset = read_json(os.path.join(f"data/datasets/{dataset_name}.json"))
        source_files = [file for pattern in cfg_dataset["source_files"] for file in glob.glob(pattern)]
        files = [(path, os.path.join(cfg_dataset["dataset_path"], hash(path) + '.gz'))
                 for path in source_files]
        build(files, cfg_dataset['dmgr'], config, name=os.path.basename(cfg_dataset["dataset_path"]))

def hash(s):
    return hashlib.sha1(
        unicodedata.normalize('NFD', s).encode('utf-8')
    ).hexdigest()


def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def list_source_files(dmgrs=['TEXT_BOOK', 'TEXT_BOOK_LOW']):
    source_file_patterns = [pattern for dataset in dmgrs for pattern in
                          read_json(f"data/datasets/{dataset}.json")["source_files"]]
    source_files = [path for pattern in source_file_patterns for path in glob.glob(pattern)]
    source_files = list(set(source_files))
    source_files.sort(key=hash)
    return source_files
