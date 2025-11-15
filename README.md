# seq2seq-arabic-dialect-lemmatization

This repository contains the code and resources associated with the paper on **Lemmatizing Dialectal Arabic with Sequence-to-Sequence Models**.  

## Available Models

This paper evaluates a large number of models for dialectal Arabic lemmatization.  
We only release the **top-performing dialectal lemmatization models** covering:

- Egyptian Arabic (EGY)  
- Gulf Arabic (GLF)  
- Levantine Arabic (LEV)

All other models in the paper follow **the exact same pipeline and evaluation framework**; the only difference is **which model is loaded at runtime**.  
If you need access to any of the additional models reported in the paper, please contact:

**Mostafa Saeed** – <mostafa.saeed@nyu.edu>

## Environment Setup

We recommend using Conda to manage the environment.

```bash
git clone https://github.com/CAMeL-Lab/seq2seq-arabic-dialect-lemmatization.git
cd seq2seq-arabic-dialect-lemmatization

# 1. Create a new conda environment
conda create -n dialectal_lemma_env python=3.12.7

# 2. Activate the environment
conda activate dialectal_lemma_env

# 3. Install Python dependencies
pip install -r requirements.txt
```

## Setting Up the Egyptian Morphological Database and Licensed Datasets

To run any of the top-performing dialectal lemmatizers for EGY, GLF, or LEV , you must first obtain the Egyptian CALIMA morphological database.
This resource cannot be distributed directly due to licensing constraints, but can be reconstructed locally using [Muddler](https://github.com/CAMeL-Lab/muddler).

1. The Egyptian morph_db is derived from the [LDC Standard Arabic Morphological Analyzer (SAMA) Version 3.1](https://catalog.ldc.upenn.edu/LDC2010L01)  and then run the following:
```bash
muddler unmuddle \
  -s "LDC2010L01.tgz source" \
  -m "data and morph DB/EGY data/arz_magold_dev.muddle" \
  "data and morph DB/EGY data/arz_magold_dev.csv"
```

2. The Egyptian dataset used in this paper is derived from the licensed resource: [BOLT Egyptian Arabic Treebank](https://catalog.ldc.upenn.edu/LDC2018T23).
   After obtaining ```bolt_arz-df_LDC2018T23.tgz```, you can reconstruct the train, dev, and test sets using the encrypted .muddle files included in this repository by running the following.

#### For the Training Set
```bash
muddler unmuddle \
  -s "bolt_arz-df_LDC2018T23.tgz source" \
  -m "data and morph DB/EGY data/arz_magold_train.muddle" \           
  "data and morph DB/EGY data/arz_magold_train.csv"
```

#### For the Development Set
```bash
muddler unmuddle \
  -s "bolt_arz-df_LDC2018T23.tgz source" \
  -m "data and morph DB/EGY data/arz_magold_dev.muddle" \           
  "data and morph DB/EGY data/arz_magold_dev.csv"
```

#### For the Test Set
```bash
muddler unmuddle \
  -s "bolt_arz-df_LDC2018T23.tgz source" \
  -m "data and morph DB/EGY data/arz_magold_test.muddle" \           
  "data and morph DB/EGY data/arz_magold_test.csv"
```


## Running the Best Dialectal Lemmatizer

After preparing the required datasets and morphological databases, you can run the best-performing lemmatizer using:
```bash
python run_best_dialectal_lemmatizer.py \
    --dialect glf \
    --data_path data/GLF/gumar_dev.csv \
    --output glf_predictions.csv
```

| Argument      | Description                                               |
| ------------- | --------------------------------------------------------- |
| `--dialect`   | Select the target dialect. Options: `{egy, glf, lev}`     |
| `--data_path` | Path to the input dataset (CSV file for EGY, GLF, or LEV) |
| `--output`    | Path where the predicted lemmas will be saved             |

# Citation
If you use this repository, models, or datasets in your research, please cite:

```bibtex
@inproceedings{saeed-habash-2025-lemmatizing,
    title     = "Lemmatizing Dialectal {A}rabic with Sequence-to-Sequence Models",
    author    = "Saeed, Mostafa  and Habash, Nizar",
    booktitle = "Proceedings of The Third Arabic Natural Language Processing Conference",
    month     = nov,
    year      = "2025",
    address   = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    abstract  = "Lemmatization for dialectal Arabic poses many challenges due to the lack of orthographic standards and limited morphological analyzers. This work explores the effectiveness of Seq2Seq models for lemmatizing dialectal Arabic, both without analyzers and with their integration. We assess how well these models generalize across dialects and benefit from related varieties. Focusing on Egyptian, Gulf, and Levantine dialects with varying resource levels, our analysis highlights both the potential and limitations of data-driven approaches. The proposed method achieves significant gains over baselines, performing well in both low-resource and dialect-rich scenarios."
}
```
