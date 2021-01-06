POS_snu_tta = {
    'NNG': 'NNG',
    'NNP': 'NNP',
    'NNB': 'NNB',
    'NNM': 'NNB',
    'NP': 'NP',
    'NR': 'NR',
    'VV': 'VV',
    'VA': 'VA',
    'VXV': 'VX',
    'VXA': 'VX',
    'VX': 'VX',
    'VCP': 'VCP',
    'VCN': 'VCN',
    'MDT': 'MMA',
    'MDN': 'MMN',
    'MAG': 'MAG',
    'MAC': 'MAJ',
    'IC': 'IC',
    'JKS': 'JKS',
    'JKC': 'JKC',
    'JKG': 'JKG',
    'JKO': 'JKO',
    'JKM': 'JKB',
    'JKI': 'JKV',
    'JKQ': 'JKQ',
    'JX': 'JX',
    'JC': 'JC',
    'EPH': 'EP',
    'EPT': 'EP',
    'EPP': 'EP',
    'EFN': 'EF',
    'EFQ': 'EF',
    'EFO': 'EF',
    'EFA': 'EF',
    'EFI': 'EF',
    'EFR': 'EF',
    'ECE': 'EC',
    'ECD': 'EC',
    'ECS': 'EC',
    'ETN': 'ETN',
    'ETD': 'ETM',
    'XPN': 'XPN',
    'XPV': 'XPN',
    'XSN': 'XSN',
    'XSV': 'XSV',
    'XSA': 'XSA',
    'XSM': 'XSA',
    'XSO': 'XSA',
    'XR': 'XR',
    'SF': 'SF',
    'SP': 'SP',
    'SS': 'SS',
    'SE': 'SE',
    'SO': 'SO',
    'SW': 'SW',
    'OL': 'SL',
    'OH': 'SH',
    'ON': 'SN',
    'UN': 'NA',
    'UV': 'NA',
    'UE': 'NA',
    'EMO': 'NA'
}



def legacy_sentences_from_raw_text(path, limit=None, force=False):
    pkl_path = path + '.' + str(limit or 'all') + ".pkl"
    if glob.glob(pkl_path) and (not force):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            tokens = data['tokens']
            text = data['text']
    else:
        print("POS tagging...")
        print("WIP")
        # kkma = Kkma()
        from .preprocessing import legacy_preprocess
        with open(path, "r", encoding="utf-8") as reader:
            text = reader.readlines()
            if limit: text = text[0:limit]
            text = legacy_preprocess.__func__(''.join(text))

            # text = kkma.sentences(text)
            # tokens = [kkma.pos(x) for x in text]
            import constant
            print("WIP")
            from tokenizer import TokenizerSpm #FIXME : get tokenizer from args
            print("")
            print(constant.TOKENIZER_DIR)
            tokenizer = TokenizerSpm(
                constant.TOKENIZER_DIR,
                train_args=None
            )
            tokenizer.load(enable_tf=False)
            text = text.split('. ')
            text = [x+'.' for x in text[:-1]]+[text[-1]]
            #print(text)
            tokens = [tokenizer.tokenize(x) for x in text]
            #print(tokens)
            print("Need to add < s > < /s > [UNK] [CLS] ...")
            with open(pkl_path, "wb") as f:
                pickle.dump({'tokens': tokens, 'text': text}, f)
    return tokens, text
