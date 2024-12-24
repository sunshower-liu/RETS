# RETS
Materials for the paper "Relation Also Knows: Rethinking the Recall and Editing of Factual Associations in Auto-Regressive Transformer Language Models".

## Folders
The *code* folder contains the codes for RETS editing. The *knowledge_detection* folder contains the codes for the detection of relational knowledge. The *dataset* folder contains the supplemented counterfact dataset with R-Specificity samples. The *evaluation* folder contains codes for evaluation supplemented with the R-Specificity criteria.

## Set up
```bash
pip install -r requirements.txt
```
For editing samples, see the tutorial in *code*. For evaluation, see the tutorial in *evaluation*.

## Acknowledgement
Our code is based on [ROME](https://github.com/kmeng01/rome) and [Dissecting_factual_predictions](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions).