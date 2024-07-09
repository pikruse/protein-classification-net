# Protein Work

## Task 1: 
*given protein structure input graph (.pdb file), build a deep neural network to classify the input’s halide specificity between chloride or bromide (multi-class classification)*

## Task 1 Steps:

1. Required reading
   a. how do DL researchers treat proteins
2. Get the .pdb files and corresponding .fasta files onto DGX from protein data bank
3. Data cleaning
   a. Removing duplicates / Filter similar proteins
   b. Identify non-specific cases (binds to both chloride and bromide separately)
   c. Identify proteins that have 2+ pockets specific to chloride or bromide
4. Data pre-processing
   a. possibly extract features
   b. convert .pdb files into pytorch dataloader format
5. Model development
   a. Based on the data we have, create a model (LLM, GCN)
6. train the model
   a. test and evaluate 
   b. Scalability
7. Iteration

## Task 2:
*build a deep neural network to generate/modify proteins s.t. they can bind to either chloride or bromide*

## Task 2 Steps:
