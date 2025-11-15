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
