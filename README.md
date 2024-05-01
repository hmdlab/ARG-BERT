# ARG-BERT
The repository contains an implementation of ARG-BERT, a BERT model that predicts the resistance mechanism of antibiotic resistance genes. If you use this model, please cite our paper. If you have any problems or comments about this repository, please contact us.
## 1. Dependencies
We have tested the model in the following environments.
TBA
Install the necessary packages by running
TBA
## 2. Fine-tuning
Run fine-tuning.py to train the ProteinBERT on ARGs by running the follwing commands.
If you train on the HMD-ARG DB, run:
python3 finetuning.py \
--fold FOLD \
--gpu GPU \
--seed SEED
FOLD, GPU and SEED are integers of type int, indicating the number of iterations in 5-fold CV, the GPU device you will use and the random seed respectively.

Or if you train on the Low Homology Dataset, run:
python3 finetuning.py \
--fold FOLD \
--use_LHD \
--threshold THRESHOLD
--gpu GPU \
--seed SEED
THRESHOLD is the floating point numbers of type float, indicating the sequence similarity thresholds set when creating the LHD.

## 3. Test
## 4. Attention Analysis
## 5. Citation
