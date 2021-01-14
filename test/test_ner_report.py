
# df = pd.DataFrame({'train':loss.keys(),'loss':loss.values()}); df.plot.scatter(x='train',y='loss',marker='.'); plt.show()

# with open("data/incoming/EBOOK/의료윤리.txt", 'r') as fp: lines = fp.readlines()
# person = []
# for line in lines:
#  recog = ner.bert_ner.recognize(line,config)
#  #print(len(recog['word']), recog['tag'], line)
#  a = zip(recog['word'], recog['tag'])
#  for x in a:
#   if x[1] == 'PER':
#    person.append(x[0])
#    print(len(recog['word']), recog['tag'], line)
#  # print(list(a))
