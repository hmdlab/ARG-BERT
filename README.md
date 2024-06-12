# ARG-BERT
The repository contains an implementation of ARG-BERT, a BERT model that predicts the resistance mechanism of antibiotic resistance genes. If you use this model, please cite our paper. If you have any problems or comments about this repository, please contact us.
## 1. Dependencies
We have tested the model in the following environments.
```
Linux: x86_64
OS: Ubuntu 20.04.6
GPU: NVIDIA A100 80G
CUDA Version: 11.6
Nvidia Driver Version: 510.47.03
```
We have all the necessary packages (see Dockerfile) in the Docker environment. Install them with the following command:
```
docker build -t USERNAME/CONTAINERNAME --build-args port=PORT --build-args password=PASSWORD .
docker login && docker push USERNAME/CONTAINERNAME
docker pull USERNAME/CONTAINERNAME
docker run -p PORT:PORT -e -it --gpus all --rm -v $PWD:/home USERNAME/CONTAINERNAME
```
## 2. Dataset and Fine-tuning
### 2.1 Dataset
Sorry we cannot publish the HMD-ARG DB and Low Homology Dataset, but the format of the data is shown in the sample data in `Sample_data`.

We saved the larger files, such as the output results and the Attention values for all sequences in the HMD-ARG DB, at [https://waseda.box.com/v/ARG-BERT-suppl](https://waseda.box.com/v/ARG-BERT-suppl).
You could run the script if you stored the `Prediction results` in `Analysis_and_Figures` in this repository and the contents of the `Attention_analysis` in a directory of the same name in `Analysis_and_Figures`.

### 2.2 Fine-tuning
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

To get the input sequences' attention, use the command `--get_all_attention` or `-attn` in both cases.

## 4. Attention Analysis
The code for the analysis is available in a Jupyter notebook in `Analysis_and_Figures/Attention_analysis`. 

In `Fig3.ipynb`, the experiments described in sections 2.6.1 and 3.2.1 of our paper are carried out, and Figure 3 and Figure S5 can be output.

`Fig4.ipynb` and `Fig5.ipynb` also carry out the experiments described in sections 2.6.2 and 3.2.2, and can output Figure 4ac, S6a and S7a for `Fig4.ipynb` and Figure 5 for `Fig5.ipynb`.

## 5. License
We used ProteinBERT licensed under the MIT license; the copyright notice and permission notice for ProteinBERT are given here.

```
Copyright (C) 2022 Nadav Brandes

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE X CONSORTIUM BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of <copyright holders> shall not be used in advertising or otherwise to promote the sale, use or other dealings in this Software without prior written authorization from <copyright holders>.
```

## 6. Citation
If you would like to use ARG-BERT, please cite our paper
```
@article{yagimoto2024prediction,
  title={Prediction of antibiotic resistance mechanisms using a protein language model},
  author={Yagimoto, Kanami and Hosoda, Shion and Sato, Miwa and Hamada, Michiaki},
  journal={bioRxiv},
  pages={2024--05},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
