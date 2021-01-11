import pandas as pd
import numpy as np
import os

def build(_, modules, cfg_dataset, config):
    """

    Args:
        formatter:
        tokenizer:

    Returns:

    """
    tokenizer = modules.tokenizers[cfg_dataset["tokenizer"]]
    dfs = []
    file_intermediate = os.path.join("data/datasets/NER/intermediate.csv")
    file_indexed = os.path.join("data/datasets/NER/indexed.pkl.gz")
    file_processed = os.path.join("data/datasets/NER/processed.pkl.gz")

    if not os.path.isfile(file_intermediate):
        for cfg in cfg_dataset["config"]:
            df = pd.read_csv(**cfg['pd_read_csv_kwargs'])

            ## XXX
            ## Python 3.8.5, [GCC 9.3.0] on linux, pd.__version__ == '1.2.0', np.__version__ == '1.19.4'
            defs = [x for x in cfg['command'] if x.startswith('def ')]
            cmds = [x for x in cfg['command'] if not x.startswith('def ')]
            for x in defs :
                exec(x, globals())
            exec('def command(df):'+';'.join(cmds+["return df"]), globals())
            df = command(df)
            ## /XXX

            if 'subword' not in df.columns:
                df['subword'] = df['word']
            if 'idx_subword' not in df.columns:
                df['idx_subword'] = np.zeros(df.shape[0],dtype=int)

            tags = cfg['tags']
            # df['pos_orig'] = df['pos']
            df = df.assign(pos=df['pos'].apply(lambda x : (str(tags[x]) if (x in tags) else 'O')))
            dfs.append(df)
        df = pd.concat(dfs)
        df['idx_sentence'] = df['idx_sentence'].diff().apply(lambda x: 0 if x == 0 else 1).cumsum() # Reindex idx_sentence
        df.to_csv(file_intermediate)

    if not os.path.isfile(file_indexed):
        df = pd.read_csv(file_intermediate, index_col=0)

        df_tmp = df.set_index(['idx_sentence'])
        df_tmp['word'] = np.where(df_tmp['idx_subword']==0, df_tmp['word'], '') # Remove duplicate word from subword set
        df_tmp = df_tmp.groupby(level=0, axis=0).agg({ # Apply list by group
            'word': lambda x : ' '.join([str(y) for y in x]),
            'subword': lambda x: x.tolist(),
            'pos': lambda x: x.tolist(),
            'pos_orig': lambda x: x.tolist()
        })
        # Process whitespace. hacky..
        df_tmp['word'] = df_tmp['word'].apply(lambda text : text.replace(' ', '').replace('_', ' ') if (' _ ' in text) else text)

        df_tmp["ner_pos"] = [[(a,b) for a,b in zip(x,y)] for x,y in zip(df_tmp["subword"].tolist(),df_tmp["pos"].tolist())]
        df = df_tmp[['word','ner_pos']]
        df.to_pickle(file_indexed,compression='infer')

    if not os.path.isfile(file_processed):
        df = pd.read_pickle(file_indexed, compression='infer')
        result = []
        for index, row in df.iterrows():
            my_tokenization = tokenizer.tokenize(row['word'], config=config)["text"]
            their_tokenization = row['ner_pos']
            # Assume : my_tokenization ~ my_tokenzer(their_tokenization)
            tagged_tokens = ([(m,their_token_tagged[1])
                              for their_token_tagged in row['ner_pos']
                              for m in tokenizer.tokenize(str(their_token_tagged[0]), config=config)["text"]
                              ])

            tokens = [x[0] for x in tagged_tokens]
            ner_tags = [x[1] for x in tagged_tokens]
            result.append((tokens,ner_tags))

        df = pd.DataFrame(result, columns=["tokens","tags"])
        df.to_pickle(file_processed,compression='infer')

    pd.read_pickle(file_processed, compression='infer')
