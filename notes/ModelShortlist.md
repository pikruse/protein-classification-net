# Protein Foundation Model Shortlist
Since our dataset is smaller than previously anticipated, it is best to use a foundation model that has been pre-trained on similar data. Here is a shortlist of candidates for our base model:

## 1. [`GPTProteinPretrained`](https://huggingface.co/datasets/lamm-mit/GPTProteinPretrained)
* This model is an autoregressive transformer (like ChatGPT). This means it can predict take in and predict protein sequences (might be useful for enzyme engineering too). 

* The [pre-training dataset](https://huggingface.co/datasets/lamm-mit/GPTProteinPretrained) seems to merely contain strings of **amino acid sequences**, which I believe we can extract from the `.pdb` files we have.

## 2. [`SaProt`](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v1)
* SaProt is a type of protein language model, similar to the ESM family. The key difference is that SaProt also incorporates structural information about the protein, using an integrated structure-residue vocabulary.

* The output seems to operate similarly to the ESM model, performing masked protein prediction, which can be done with various types of supervision.

* The in`