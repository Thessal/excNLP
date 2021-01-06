import glob
import os
import hashlib
import json
import modules
import dmgr.text

dmgrs = {"text":dmgr.text}

def build(files, cfg_dataset, config, name="dmgr"):
    print(f'{len(files)} files : {name}')
    if cfg_dataset["dmgr"] == "text":
        dmgr.text.build(files,
                        modules.formatters[cfg_dataset["formatter"]],
                        modules.tokenizers[cfg_dataset["tokenizer"]],
                        config)


def build_all(config):
    cfg_dataset = read_json(os.path.join("data/datasets/TEXT_BOOK.json"))
    files = [(path, os.path.join(cfg_dataset["dataset_path"], hash(path) + '.gz'))
             for path in glob.glob(cfg_dataset["source_files"])]
    build(files, cfg_dataset['dmgr'], config, name=os.path.basename(cfg_dataset["dataset_path"]))

def hash(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data

