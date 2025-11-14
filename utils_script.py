from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
import pandas as pd

def load_disambiguators(dialect, back_off='NOAN_PROP'):
    if dialect == 'egy':
        egy_db = MorphologyDB('morphological DBs/calima-egy-c044_0.1.1.utf8.db')
        egy_calima_analyzer = Analyzer(egy_db, back_off)

        egy_bert = BERTUnfactoredDisambiguator.pretrained(
            'egy', top=5000, pretrained_cache=False
        )
        egy_bert._analyzer = egy_calima_analyzer
        disambig_list = [egy_bert]

    elif dialect == 'glf':
        egy_db = MorphologyDB('morphological DBs/calima-egy-c044_0.1.1.utf8.db')
        egy_calima_analyzer = Analyzer(egy_db, back_off)

        egy_bert = BERTUnfactoredDisambiguator.pretrained(
            'egy', top=5000, pretrained_cache=False
        )
        egy_bert._analyzer = egy_calima_analyzer
        disambig_list = [egy_bert]

        glf_bert = BERTUnfactoredDisambiguator.pretrained(
            'glf', top=5000, pretrained_cache=False
        )
        glf_bert._analyzer._backoff = back_off

        lev_bert = BERTUnfactoredDisambiguator.pretrained(
            'lev', top=5000, pretrained_cache=False
        )
        lev_bert._analyzer._backoff = back_off
        disambig_list = [glf_bert, egy_bert, lev_bert]

    elif dialect == 'lev':
        egy_db = MorphologyDB('morphological DBs/calima-egy-c044_0.1.1.utf8.db')
        egy_calima_analyzer = Analyzer(egy_db, back_off)

        egy_bert = BERTUnfactoredDisambiguator.pretrained(
            'egy', top=5000, pretrained_cache=False
        )
        egy_bert._analyzer = egy_calima_analyzer
        disambig_list = [egy_bert]

        glf_bert = BERTUnfactoredDisambiguator.pretrained(
            'glf', top=5000, pretrained_cache=False
        )
        glf_bert._analyzer._backoff = back_off

        lev_bert = BERTUnfactoredDisambiguator.pretrained(
            'lev', top=5000, pretrained_cache=False
        )
        lev_bert._analyzer._backoff = back_off
        disambig_list = [lev_bert, egy_bert, glf_bert]
    
    return disambig_list

def remove_sukun_and_tatweel(text):
    if isinstance(text, str):
        return text.replace('\u0652', '').replace('\u0640', '')
    return text

def evaluate_disambiguation_with_sentences(
    df,
    disambig_list):
    # Step 1: Prepare list of tokenized sentences
    sentences_list = []
    indices_list = []
    # Group and sort
    for _, group in df.groupby('sentence_index'):
        group = group.sort_values('word_index')
        sentences_list.append(list(group['word'].astype(str)))
        indices_list.append(group.index.tolist())

    # Step 2: Disambiguate with each disambiguator
    all_flattened_analyses = []

    for i, disambig in enumerate(disambig_list):
        results = disambig.disambiguate_sentences(sentences_list)

        for sentence_idx, sentence in enumerate(results):
            indices = indices_list[sentence_idx]

            for word_idx, word in enumerate(sentence):
                word_text = word.word
                original_idx = indices[word_idx]

                for analysis in word.analyses:
                    entry = {
                        'disambig_model': disambig,
                        'sentence_index': sentence_idx,
                        'word_index': word_idx,
                        'word': word_text,
                        'original_index': original_idx,
                        'diac': analysis.diac,
                        'score': analysis.score
                    }
                    entry.update(analysis.analysis)
                    all_flattened_analyses.append(entry)

    # Step 3: Create combined DataFrame
    disambig_df = pd.DataFrame(all_flattened_analyses)

    disambig_df = disambig_df.drop_duplicates(
        subset=['sentence_index', 'word_index', 'lex', 'pos', 'stemgloss']
    ).reset_index(drop=True)
    
    counts = disambig_df.groupby(['sentence_index', 'word_index'])['score'].transform('count')
    
    # Apply filtering condition
    disambig_df = disambig_df[
        (disambig_df['score'] == 1) |
        ((counts == 1) & (disambig_df['score'] == 0))
    ].reset_index(drop=True)

    # Step 2: Create s2s_lookup once
    s2s_lookup = df.set_index(['sentence_index', 'word_index'])['s2s_predicted_lemma'].to_dict()

    # Step 3: Process each word individually (like code 1)
    selected_rows = []

    grouped = disambig_df.groupby(['sentence_index', 'word_index'])

    for (s_idx, w_idx), group in grouped:
        # print(s2s_lookup.get((s_idx, w_idx), ''))
        # print(s_idx)
        # print(w_idx)
        val = s2s_lookup.get((s_idx, w_idx), '')
        predicted_lemma = str(val).strip() if not pd.isna(val) else ''

        # Look for exact match on lex
        match_idx = group[group['lex'] == predicted_lemma].index

        if not match_idx.empty:
            selected = group.loc[match_idx[0]]
        else:
            selected = group.iloc[0]

        selected_rows.append(selected)

    top1_df = pd.DataFrame(selected_rows).reset_index(drop=True)
        
    # Step 5: Merge predictions with gold labels
    df['lex'] = top1_df['lex']
    df['pos'] = top1_df['pos']
    df['stemgloss'] = top1_df['stemgloss']  
    df['lex'] = df['lex'].apply(remove_sukun_and_tatweel)

    mask = df['pos'].isin(['punc', 'digit'])
    df.loc[mask, 's2s_predicted_lemma'] = df.loc[mask, 'word']
    
    return df
