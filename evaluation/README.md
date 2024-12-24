This folder is used for the evaluation on the supplemented COUNTERFACT dataset with the R-Specificity criteria. 
# Set Up
Before evaluation, you need to copy *counterfact_rs.json* from "*../dataset*" to "*../code/data*" and run *evaluation.py* under "*../code*".

# Run
An example for running evaluation is shown as below.
```bash
python -m evaluation.evaluate_rs --alg_name=RETS --model_name=gpt2-xl --hparams_fname=gpt2-xl.json
```