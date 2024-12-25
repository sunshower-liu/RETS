This folder is used for the evaluation on the supplemented COUNTERFACT dataset with the R-Specificity criteria. 
# Set Up
Before evaluation, you need to copy *counterfact_rs.json* from "*../dataset*" to "*../code/data*" and run *evaluation.py* under "*../code*".

# Run
An example for running evaluation is shown as below. Results are saved in "*../code/results*". 
```bash
python -m evaluation.evaluate_rs --alg_name=RETS --model_name=gpt2-xl --hparams_fname=gpt2-xl.json
```
To summarize, run summarize.py for computing the final scores.
```bash
python -m evaluation.summarize --dir_name=RETS --runs=run_000 --first_n_cases 1000
```