## Batch run paraphrasing API

import glob


def is_hangul(text, ratio=0.6):
    text = text.replace(' ', '')
    text = text.replace('...', '.')
    text = text.replace('!!!', '!')
    return sum([(ord('가') <= ord(x) and ord(x) <= ord('힣')) for x in text]) > len(text) * ratio


files = glob.glob("../proprietary/data/TRANSFER/Raw/mort/*.txt")
# NOTE : dataset mort contains lots of \", so remove it

lines = []
for file in files:
    with open(file, 'r', encoding='utf-8') as fp:
        lines.extend(fp.readlines())

orig = []
trans = []
buffer = {"orig": [], "trans": []}
state = None
for line in lines:
    assert (len(orig) == len(trans))
    line = line.strip()
    line = line.strip(chr(65279))
    if state == "orig" and line:
        buffer["orig"].append(line)
        state = None
    elif state == "trans" and line:
        buffer["trans"].append(line)
        state = None
    if line.lower().startswith("/s"):
        state = 'orig'
    elif line.lower().startswith("/t"):
        state = 'trans'
    elif line.lower().startswith("/e"):
        if buffer["orig"][0] == "/EH0/IH0/ax/ae0": continue
        if (len(buffer["orig"]) == 0) or (len(buffer["trans"]) == 0):
            print(buffer)
            raise ValueError
        buffer["orig"] = " ".join(buffer["orig"])
        buffer["trans"] = " ".join(buffer["trans"])
        if is_hangul(buffer["trans"]):
            orig.append(buffer["orig"])
            trans.append(buffer["trans"])
        else:
            print(f"Discarding {buffer}")
        buffer = {"orig": [], "trans": []}
        state = None

print(len(orig))
pairs = zip(orig, trans)
import pandas as pd
import secret

df = pd.DataFrame(columns=['orig', 'paraphrased'])
for i, pair in enumerate(list(pairs)):
    #if i < 208 : continue
    print(i)
    try:
        trans = secret.paraphrase(pair[0])
        if is_hangul(trans):
            df = df.append({'orig': pair[1], 'paraphrased': trans}, ignore_index=True)
            print(pair[1], '___', trans)
        else:
            print(f"Discarding {trans}")
    except Exception as e:
        print(f"Error processing {pair[0]}")
        print(e)
        pass

df.to_csv('../proprietary/data/TRANSFER/mort/paraphrase.csv', index=False)