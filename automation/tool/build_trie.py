# Usage : cd .. && python3.8 -m automation.tool.build_trie
# TODO : merge it into trie.py
from ..document_model.trie import Trie
import glob
#import cPickle as pickle
import pickle


def save_pkl(filename, data):
    with open('automation/proprietary/data/POS/' + filename + '.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(filename):
    with open('automation/proprietary/data/POS/' + filename + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    return b


def raw_words(paths=None):
    files = sum([list(glob.glob(path)) for path in paths], [])
    print(f"{len(files)} files loaded.")
    count = 0
    for file in files[:100]:
        count = count + 1 
        if count%1000==0 : print(count)
        try :
            with open(file, "r", encoding="utf-8") as fp:
                for line in fp.readlines():
                    if not line.strip(): pass
                    words = line.strip().split(' ')
                    for word in words:
                        word = word.strip()
                        if word: yield word
        except Exception as e:
            print(e)
            print(file)
            #break

trie = Trie()
path = ["automation/proprietary/data/TEXT/Raw/EBOOK/*.txt",
        "automation/proprietary/data/TEXT/Raw/WEB/*.txt",
        "automation/proprietary/data/TEXT/Raw/EBOOK_ETC/*/*.txt",
        ]  # 7GBs of texts
for word in raw_words(paths=path):
    trie.insert(word)
print(trie.top_n_items(n=10))
print(trie.query('')[:10])
print(trie.query('ㅇ')[:10])
print(trie.root.children)
save_pkl('Trie', trie.query(''))
