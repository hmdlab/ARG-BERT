# ARG-BERT
The repository contains an implementation of ARG-BERT, a BERT model that predicts the resistance mechanism of antibiotic resistance genes. If you use this model, please cite our paper. If you have any problems or comments about this repository, please contact us.
## 1. Dependencies
We have tested the model in the following environments.
TBA
Install the necessary packages by running
TBA
## 2. Fine-tuning
Run `finetuning.py` to train the ProteinBERT on ARGs by running the follwing commands.
If you would like to train with the HMD-ARG DB, run:
```
python3 finetuning.py \
--fold FOLD \
--gpu GPU \
--seed SEED
```
FOLD, GPU and SEED are integers of type int, indicating the number of iterations in 5-fold CV, the GPU device you will use and the random seed respectively.

Alternatively, if you would like to train with the Low Homology Dataset, run:
```
python3 finetuning.py \
--fold FOLD \
--use_LHD \
--threshold THRESHOLD
--gpu GPU \
--seed SEED
```
THRESHOLD is the floating point numbers of type float, indicating the sequence similarity thresholds set when creating the LHD.

## 3. Test
Run `test.py` by running the follwing commands.
If you would like to test with the HMD-ARG DB, run:
```
python3 test.py \
--fold FOLD \
--seed SEED
```

Alternatively, if you would like to test with the Low Homology Dataset, run:
```
python3 finetuning.py \
--fold FOLD \
--use_LHD \
--threshold THRESHOLD
--gpu GPU \
--seed SEED
```

To get the input sequences' attention, use command `--get_all_attention` or `-attn` in both cases.

## 4. Attention Analysis
The code for the analysis is available in a Jupyter notebook in `Attention_analysis`. 

In `Calculate amino acid conservation score.ipynb`, the experiments described in sections 2.6.1 and 3.2.1 of our paper are carried out, and Figure 3 and Figure S5 can be output.

`Identify Attention-intensive regions.ipynb` and `GO enrichment analysis.ipynb` also carry out the experiments described in sections 2.6.2 and 3.2.2, and can output Figure 4ac, S6a and S7a for `Identify Attention-intensive regions.ipynb` and Figure 5 for `GO enrichment analysis.ipynb`.

## 5. Citation
If you would like to use ARG-BERT, please cite our paper
```
@article
```
