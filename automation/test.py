from konlpy.tag import Kkma
import pickle
import glob

# def morph(input_data):  # 형태소 분석
#     kkma = Kkma()
#     preprocessed = kkma.pos(input_data)
#     print(preprocessed)
# morph("우리가 가지고 있는 수학 외적인 모든 지식과 표현 가능한 논리들은 추측으로 이루어져 있다.")

def preprocess(_text):
    def _whitespace(_text):
        t = _text.replace('\n', ' ')
        t = ' '.join(_text.split())
        return t

    # _parse_page()
    t = _whitespace(_text)
    # _unknown_characters()
    return t

def read_raw_text(path, limit=None, force=False):
    pkl_path = path + '.' + str(limit or '') + ".pkl"
    if glob.glob(pkl_path) or force:
        with open(pkl_path, "rb") as f:
            tokens = pickle.load(f)
    else:
        kkma = Kkma()
        with open(path, "r", encoding="utf-8") as reader:
            text = reader.readlines()
            if limit: text = text[0:limit]
            text = preprocess(''.join(text))
            tokens = kkma.pos(text)
            with open(pkl_path, "wb") as f:
                pickle.dump(tokens, f)
    return tokens


def _check_tags(path):
    x21_dic = glob.glob(path)
    tags_all = set()
    tags_comment = []
    for path in x21_dic :
        with open(path, 'r', encoding="utf-8") as reader:
            text = reader.readlines()
            tags_comment += ([x.replace('\n', '') for x in text if (x[0] == '/' and x[3] != '=' and x.rstrip()[-1] == ')')])
            tags = set([x.split('/')[1].split(';')[0].split(':')[0].split(']')[0].split('+')[0].replace('\n','') for x in text if (len(x.split('/'))>1 and x[0]!='/')])
            tags_all = tags_all.union(tags)
    print('\n'.join(sorted(set(tags_comment))))
    return tags_all


print()
print("2.1")
x21_tags = _check_tags("proprietary/kkma/2.1/dic/*.dic")
print(sorted(x21_tags))
print(len(x21_tags))
print()
print("2.0")
x20_tags = _check_tags("proprietary/kkma/kkma/dic/*.dic")
print(sorted(x20_tags))
print(len(x20_tags))
print()
x21_manual = ['NNG','NNP','NNB','NNM','NR','NP','VV','VA','VXV','VXA','VCP','VCN','MDT','MDN','MAG','MAC','IC','JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JX','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','XSM','XSO','XR','SF','SP','SS','SE','SO','SW','UN','UV','UE','OL','OH','ON']
print([x for x in x21_manual if x not in x21_tags])
print([x for x in x21_tags if x not in x21_manual])
print()
print("TTAK.KO-11.0010/R1")
# x11_0010_R1_tags = ['XSA', 'XPV', 'SW', 'JKG', 'JKM', 'XPN', 'SE', 'JC', 'SO', 'VA', 'VCP', 'MD', 'NR', 'EC', 'NP', 'ETN', 'VX', 'N/A', 'XSM', 'NNA', 'SF', 'VV', 'JKO', 'JX', 'MAC', 'VCN', 'ETD', 'XSV', 'JKQ', 'EF', 'XSO', 'MAG', 'XSN', 'SS', 'IC', 'NNB', 'EP', 'SP', 'XR', 'JKS', 'JKI', 'JKC']
x11_0010_R1_tags = ['NNG','NNP','NNB','NP','NR','VV','VA','VX','VCP','VCN','MMA','MMD','MMN','MAG','MAJ','IC','JKS','JKC','JKG','JKO','JKB','JKV','JKQ','JX','JC','EP','EF','EC','ETN','ETM','XPN','XSN','XSV','XSA','XR','SF','SP','SS','SE','SO','SW','SL','SH','SN','NA']
print(sorted(x11_0010_R1_tags))
print(len(x11_0010_R1_tags))
with open('proprietary/korbert/001_bert_morp_pytorch/vocab.korean_morp.list', 'r', encoding="utf-8") as reader:
    text = reader.readlines()
    x11_0010_R1_tags_test = (sorted(set([x.split('_')[0].split('/')[1] for x in text if (('_' in x )and ('/' in x))])))
print(x11_0010_R1_tags_test)
print([x for x in x11_0010_R1_tags_test if x not in x11_0010_R1_tags])
print([x for x in x11_0010_R1_tags if x not in x11_0010_R1_tags_test])
print()
print("konlpy")
raw_texts = [f"proprietary/text/{x}.txt" for x in ["공정하다는 착각", "생각에 관한 생각", "시지프 신화", "의료윤리", "행복의 기원"]]
tokens = read_raw_text(raw_texts[3], limit=None)
types = (set([x[1] for x in tokens]))
print(types)
print(len(types))
print([x for x in types if x not in x20_tags])
print([x for x in x20_tags if x not in types])


