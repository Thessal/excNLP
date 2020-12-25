from glob import glob
import random
RAW_TEXT_FILES = glob("proprietary/data/TEXT/Raw/**/*.txt", recursive=True)
TOKENIZER_DIR = "proprietary/data/POS/"
TOKENIZER_DIR = "proprietary/data/POS/"
RAW_TEXT_FILES_SHUFFLE = (RAW_TEXT_FILES*6).copy()
random.shuffle(RAW_TEXT_FILES_SHUFFLE)