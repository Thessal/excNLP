# TODO : merge it into trie.py

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
path = '../'
from document import Trie

import glob
import timeit

# Utility functions
import pickle # import cPickle as pickle

def save_pkl(filename, data):
    with open(path+'proprietary/data/POS/' + filename + '.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(filename):
    with open(path+'proprietary/data/POS/' + filename + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    return b

import unicodedata
def shortenStringCJK(string, width, placeholder='..'):
    # get the length with double byte charactes
    string_len_cjk = stringLenCJK(str(string))
    # if double byte width is too big
    if string_len_cjk > width:
        # set current length and output string
        cur_len = 0
        out_string = ''
        # loop through each character
        for char in str(string):
            # set the current length if we add the character
            cur_len += 2 if unicodedata.east_asian_width(char) in "WF" else 1
            # if the new length is smaller than the output length to shorten too add the char
            if cur_len <= (width - len(placeholder)):
                out_string += char
        # return string with new width and placeholder
        return "{}{}".format(out_string, placeholder)
    else:
        return str(string)
def stringLenCJK(string):
    # return string len including double count for double width characters
    return sum(1 + (unicodedata.east_asian_width(c) in "WF") for c in string)
def formatLen(string, length):
    # returns length udpated for string with double byte characters
    # get string length normal, get string length including double byte characters
    # then subtract that from the original length
    return length - (stringLenCJK(string) - len(string))

####


def raw_words(paths=None, files=None):
    files = sum([list(glob.glob(path)) for path in paths], []) if not files else files
    print(f"{len(files)} files loaded.")
    count = 0
    for file in files[:1000]:
        count = count + 1
        if count % 1000 == 0: print(count)
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                if not line.strip(): pass
                words = line.strip().split(' ')
                for word in words:
                    word = word.strip()
                    if word: yield word


def text_to_pkl(thres=100, run=True):
    paths = [path+"proprietary/data/TEXT/Raw/EBOOK/*.txt",
             path+"proprietary/data/TEXT/Raw/WEB/*.txt",
             path+"proprietary/data/TEXT/Raw/EBOOK_ETC/*/*.txt",
             ]  # 7GBs of texts

    files = sum([list(glob.glob(path)) for path in paths], [])
    print(f"Total {len(files)} files.")
    chunksize = 1000
    indices = list(range(int(len(files) / chunksize)))
    if run:
        for x in indices:
            begin = x * chunksize
            end = min(len(files), begin + chunksize)
            trie = Trie()
            for word in raw_words(files=files[begin:end]):
                trie.insert(word)
            trie.output = []
            trie.dfs(trie.root, '', include_subword=True, count_recursive=False)
            save_pkl(f'Trie_{x}', [x for x in trie.output if x[1] > thres])
    return [f'Trie_{x}' for x in indices]


def pkl_to_trie(pkl_files):
    trie = Trie()
    for x in pkl_files:
        b = load_pkl(x)
        for c in b:
            trie.insert(c[0], c[1])
    trie.output = []
    trie.dfs(trie.root, '', include_subword=True, count_recursive=False)
    save_pkl(f'Trie_all', [x for x in trie.output if x[1] > 100])
    return trie

run = False
global trie
print(timeit.timeit('global pkl_files; pkl_files = text_to_pkl(run=run)', number=1, globals=globals()))
print(timeit.timeit('global trie; trie = pkl_to_trie(pkl_files)', number=1, globals=globals()))
print(trie.top_n_items(n=30))


# Basic tagging
all = trie.query('')
#len_all = all[0][1]

import numpy as np
def prefix_info(input):
    x = trie.query(input)
    # TF : prefix frequency
    # IDF : number of postfix where prefix appears
    # N : average postfix count : heuristic : 50
    N = 50
    tf = x[0][1] / all[0][1]
    #tf = np.log(1+tf)
    idf = -np.log(len(x)/N)
    tfidf = tf*idf
    term_pprint = shortenStringCJK(input.ljust(20), width=15)
    print(f"Term, TF, IDF, TF-IDF : {term_pprint}\t{tf:.5f}\t{idf:.2f}\t{tfidf:.5f}")

prefix_info("재미있겠")
prefix_info("재미있게")
prefix_info("재미있ㄱ")
prefix_info("재미있")
prefix_info("재미이")
prefix_info("재미ㅇ")
prefix_info("재미")
prefix_info("재ㅁ")
prefix_info("재")
prefix_info("ㅈ")

prefix_info("섹스하고")
prefix_info("섹스하ㄱ")
prefix_info("섹스학")
prefix_info("섹스하")
prefix_info("섹스ㅎ")
prefix_info("섹스")
prefix_info("섹ㅅ")
prefix_info("섹")
prefix_info("세")
prefix_info("ㅅ")

prefix_info("싶다")
prefix_info("싶ㄷ")
prefix_info("싶")
prefix_info("시")
prefix_info("ㅅ")
#prefix_info("것")
#prefix_info("것ㅇ")
#prefix_info("재미")
#prefix_info("재미있")
#prefix_info("재미있ㅇ")
#prefix_info("재미있ㄴ")
#prefix_info("죽ㅇ")
#prefix_info("죽")
#prefix_info("섹")
#prefix_info("섹스")




