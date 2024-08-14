# Protein Foundation Model Shortlist
Since our dataset is smaller than previously anticipated, it is best to use a foundation model that has been pre-trained on similar data. Here is a shortlist of candidates for our base model:

## 1. [GPTProteinPretrained](https://huggingface.co/datasets/lamm-mit/GPTProteinPretrained)
* This model is an autoregressive transformer (like ChatGPT). This means it can predict take in and predict protein sequences (might be useful for enzyme engineering too). 

* The [pre-training dataset](https://huggingface.co/datasets/lamm-mit/GPTProteinPretrained) seems to merely contain strings of **amino acid sequences**, which I believe we can extract from the `.pdb` files we have.

## 2. [SaProt](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v1)
* SaProt is a type of protein language model, similar to the ESM family. The key difference is that SaProt also incorporates structural information about the protein, using an integrated structure-residue vocabulary.

* The output seems to operate similarly to the ESM model, performing masked protein prediction, which can be done with various types of supervision.

* The input appears similar to a normal PLM (ex: `MdEvVpQpLrVyQdYaKv`), and the `.pdb.` files should have structural information as well.

## 3. [ESM](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1)
* We can fill out an HF license application for this.

* Use instructions come from the [ESM Repository](https://github.com/evolutionaryscale/esm?tab=readme-ov-file#quickstart).

* We've already read this paper, so could be a good start.

* Model accepts partial, masked inputs across all combinations (sequence, structure tokens, SS8, SASA, Function), and returns full outputs.

## 4. [BERT Base for Proteins](https://huggingface.co/unikei/bert-base-proteins)
* This could be nice, as BERT models are trained for classification output (our initial task).

* The model is trained on human proteins, however, so I'm not sure how transferrable the domain is.

* Supports sequence intake (ex: Insulin - `MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN`).

# Other Considerations
* If we don't want to use a pre-trained model, we could venture to make a foundation model ourselves. There is plenty of data out there, but I think that this solution would be rather inefficient.

* For most of these models, we will have to modify the output s.t. we can predict whether our proteins bind to bromide or chloride. We must consider several ways to do this:
    1. Flatten the last layer and propagate to single node (will return prb. of binding to bromide or chloride)
    2. Create a token-wise classification output (highlight which region of the protein binds to the atom)
    3. Other suggestions from the experts
