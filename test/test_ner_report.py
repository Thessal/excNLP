
# df = pd.DataFrame({'train':loss.keys(),'loss':loss.values()}); df.plot.scatter(x='train',y='loss',marker='.'); plt.show()

with open("data/incoming/EBOOK/담론과 진실.txt", 'r') as fp: lines = fp.readlines()

all = []
for line in lines:
 recog = ner.bert_ner.recognize(line,config)
 #print(len(recog['word']), recog['tag'], line) a
 a = zip(recog['word'], recog['tag'])
 a = [x for x in a if (x[1]!='O' and x[1][0]!='[')]
 all.append(a)
 if a :
  print(a)
  print(line)
  print(recog['word'])
  print(recog['raw_valid_ids'])
  print(recog['raw_logits_label'])
  print("==")
  print("==")
 # for x in a:
 #  if x[1] != 'O':
 #   #person.append(x[0])
 #   print(x[0])
 #print(recog['tag'], line)
 # print(list(a))


output = {"word": words, "tag": labels, "confidence": confidence,
          "text": tokens['text'], "raw_tokens": tokens, "raw_valid_ids": valid_ids[0],
          "raw_logits_label": logits_label}