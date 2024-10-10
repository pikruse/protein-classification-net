## Next Steps for PLM Modeling
1. use ESM as a base model
2. append single neuron regression layer, representing occupancy as a decimal range [0, 1]
3. train model for regression
4. after training, evaluate predictions with 3 classes - bromide, chloride, or even. return associated occupancy to inform predictions.