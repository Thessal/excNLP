import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import pandas as pd
from document.document import Document
from formatter import simple as formatter
from tokenizer import simple as tokenizer

test = {}

doc = Document(formatter=formatter, tokenizer=tokenizer, file="./data/test.txt")
df = pd.DataFrame([line for line in doc])
df.to_pickle("./data/test_document_simple.gz", compression="gzip")
test["simple"] = df.equals(pd.read_pickle("./data/test_document_simple.gz", compression="gzip"))

print(test)