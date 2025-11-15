# main.py
import argparse
from utils import (
    load_disambiguators,
    remove_sukun_and_tatweel,
    evaluate_disambiguation_with_sentences,
    get_context_window
)

from lemma_seq2seq_inference import S2S_pred
import pandas as pd
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run Arabic disambiguation pipeline")

    parser.add_argument("--dialect", type=str, default="egy",
                        help="Dialect code (e.g., egy, lev, gla)")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input dev CSV")

    parser.add_argument("--output", type=str, default="egy_predictions.csv",
                        help="Output CSV file")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load disambiguators
    disambig_models = load_disambiguators(args.dialect, back_off='NOAN_PROP')

    # 2) Load data
    test_df = pd.read_csv(args.data_path)

    # 3) S2S prediction
    dev_name = os.path.basename(args.data_path).replace(".csv", "")

    # Sort test_df by sentence_index and word_index
    S2S_test_df = test_df.sort_values(by=["sentence_index", "word_index"])

    tqdm.pandas(desc="Generating valid context")
    S2S_test_df['input_text'] = S2S_test_df.progress_apply(
        lambda row: get_context_window(S2S_test_df, row['sentence_index'], row['word_index']), axis=1
    )

    s2s_dialect = args.dialect
    s2s_dir = S2S_pred(s2s_dialect, S2S_test_df, dev_name)
    s2s_df = pd.read_csv(s2s_dir)

    # 4) Attach predictions
    test_df["s2s_predicted_lemma"] = s2s_df["pred_top1"].astype(str)
    test_df["lex"] = s2s_df["pred_top1"].astype(str)

    # 5) Evaluate
    result_df = evaluate_disambiguation_with_sentences(test_df, disambig_models)

    # 6) Save output
    result_df[["sentence_index","word_index","word","s2s_predicted_lemma"]].to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
