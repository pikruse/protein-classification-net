# Protein Work

## Task 1: 
*given protein structure input graph (.pdb file), build a deep neural network to classify the input’s halide specificity between chloride or bromide (multi-class classification)*

## Task 1 Steps:

0. Required reading
    - how do DL researchers treat proteins
1. Get the .pdb files and corresponding .fasta files onto DGX from protein data bank
2. Data cleaning
    - Removing duplicates / Filter similar proteins
        * Exclude PDB files using following criteria: `Resolution > 2 Angstrom` and `NMR, powder diffraction, neutron diffraction`
    - Identify non-specific cases (binds to both chloride and bromide separately)
    - Identify proteins that have 2+ pockets specific to chloride or bromide
4. Data pre-processing
    - possibly extract features
    - convert .pdb files into pytorch dataloader format
5. Model development
    - Based on the data we have, create a model (LLM, GCN)
    - train the model
    - test and evaluate 
    - Scalability
6. Iteration

## Task 2:
*build a deep neural network to generate/modify proteins s.t. they can bind to either chloride or bromide*

## Task 2 Steps: