import os
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

def S2S_pred(s2s_dialect, dev_path, dev_name):
    # ---------- Load Dev Data ----------
    test_df = dev_path
    test_df['input_text'] = test_df['input_text'].astype(str)

    # ---------- Extract target word ----------
    def extract_target_word(text):
        match = re.search(r"<target>(.*?)<target>", str(text))
        return match.group(1).strip() if match else ""

    test_df["word"] = test_df["input_text"].apply(extract_target_word)

    if s2s_dialect == 'egy':
        print(f"\nRunning model: egy_s2s")
        tokenizer = T5Tokenizer.from_pretrained('CAMeL-Lab/EGY-S2S-lemmatizer', use_fast=True, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained('CAMeL-Lab/EGY-S2S-lemmatizer')
        tokenizer.add_special_tokens({'additional_special_tokens': ['<target>']})
        model.resize_token_embeddings(len(tokenizer))

    elif s2s_dialect == 'glf':
        print(f"\nRunning model: glf_s2s")
        tokenizer = T5Tokenizer.from_pretrained('CAMeL-Lab/GLF-S2S-lemmatizer', use_fast=True, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained('CAMeL-Lab/GLF-S2S-lemmatizer')
        tokenizer.add_special_tokens({'additional_special_tokens': ['<target>']})
        model.resize_token_embeddings(len(tokenizer))

    elif s2s_dialect == 'lev':
        print(f"\nRunning model: lev_s2s")
        tokenizer = T5Tokenizer.from_pretrained('CAMeL-Lab/LEV-S2S-lemmatizer', use_fast=True, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained('CAMeL-Lab/LEV-S2S-lemmatizer')
        tokenizer.add_special_tokens({'additional_special_tokens': ['<target>']})
        model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Predictions
    batch_size = 16
    all_predictions = []

    for i in tqdm(range(0, len(test_df), batch_size), desc="Generating predictions"):
        batch = test_df.iloc[i:i + batch_size]
        encodings = tokenizer(
            batch["input_text"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_length=50,
                num_beams=1,
                num_return_sequences=1,
                do_sample=False
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend(decoded_preds)

    # Evaluation
    output_df = test_df.copy()
    output_df["pred_top1"] = all_predictions
    
    output_dir = "data/s2s_output_data"
    os.makedirs(output_dir, exist_ok=True)
    # Output filename
    output_csv = os.path.join(output_dir, f"{dev_name}_{s2s_dialect}_model.csv")

    # Save
    output_df.to_csv(output_csv, index=False)

    return str(output_csv)