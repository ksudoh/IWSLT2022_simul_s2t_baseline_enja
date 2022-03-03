# IWSLT 2022 Evaluation Campaign: Simultaneous Translation Baseline (Engilsh-to-Japanese Text-to-Text)

## Table of Contents
- [IWSLT 2022 Evaluation Campaign: Simultaneous Translation Baseline (Engilsh-to-Japanese Text-to-Text)](#iwslt-2022-evaluation-campaign-simultaneous-translation-baseline-engilsh-to-japanese-text-to-text)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Setup](#setup)
    - [Clone a repository and install required packages](#clone-a-repository-and-install-required-packages)
    - [Setup fairseq (if needed)](#setup-fairseq-if-needed)
    - [(Not avaiable now) Setup fairseq for MMA-IL (if needed)](#not-avaiable-now-setup-fairseq-for-mma-il-if-needed)
    - [Setup SimulEval](#setup-simuleval)
  - [Data preparation](#data-preparation)
  - [Setting Environment Variables](#setting-environment-variables)
  - [Preprocessing](#preprocessing)
  - [ASR pretraining](#asr-pretraining)
  - [Wait-K model](#wait-k-model)
    - [Training](#training)
    - [Test](#test)
  - [(Not avaiable now) MMA-IL (Monotonic Multihead Attention with Infinite Lookback) model](#not-avaiable-now-mma-il-monotonic-multihead-attention-with-infinite-lookback-model)
    - [Training](#training-1)
---

## Requirements
- Linux-based system
  - The scripts were tested with Ubuntu 18.04 LTS but would work on 20.04 LTS
- Bash
- Python >= 3.7.0
- (CUDA; not mandatory but highly recommended)
- PyTorch (the following command installs 1.10.1 working with CUDA 11.3)
    ```shell
    $ pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    ```

## Setup
### Clone a repository and install required packages
```shell
$ git clone --recursive https://github.com/ksudoh/IWSLT2022_simul_s2t_baseline_enja.git
$ cd IWSLT2022_simul_s2t_baseline_enja
$ pip3 install -r requirements.txt
```
### Setup fairseq (if needed)
```shell
$ pushd fairseq
$ python3 setup.py build_ext --inplace
$ popd
```
### (Not avaiable now) Setup fairseq for MMA-IL (if needed)
```shell
$ pushd fairseq-mma-il/fairseq
$ python3 setup.py build_ext --inplace
$ popd
```

### Setup SimulEval
```shell
$ pushd SimulEval
$ python3 setup.py install --prefix ./
$ popd
```

## Data preparation

- Download MuST-C v2.0 and extract the package
  - Suppose you put the extracted directory `en-ja` in `/path/to/MuST-C/`.

## Setting Environment Variables
- The baseline system scripts use the following environment variables.
    - `WORKDIR` specifies the directory to be used to store the data and models.
    - You may change the setting of `TMPDIR` if you would like to use the temporary space other than `/tmp`. The scripts
```shell
$ export SRC=en
$ export TRG=ja
$ export MUSTC_ROOT=/path/to/MuST-C
$ export WORKDIR=/path/to/work
```

## Preprocessing
- The wrapper script `10-preprocess.sh` conducts the required preprocessing
```shell
$ bash ./10-preprocess.sh
```
- The wrapper script `11-prepare-eval-data.sh` prepares the test data
```shell
$ bash ./11-prepare-eval-data.sh
```

## ASR pretraining
- Set a variable `CUDA_VISIBLE_DEVICES`
- You may use multiple GPUs, but the batch size becomes larger accordingly.
```shell
$ env CUDA_VISIBLE_DEVICES=0 bash ./20-train-pretraining.sh
```

## Wait-K model
- Set variables `K` and `CUDA_VISIBLE_DEVICES`.
- You may use multiple GPUs, but the batch size becomes larger accordingly.
### Training
```shell
$ env K=20 CUDA_VISIBLE_DEVICES=0 bash ./30-train-st-wait-k.sh
```

### Test
SimulEval sometimes fails to establish the connection between the server and the client, so please terminate the process and re-run in such a case.
```shell
$ env K=20 CUDA_VISIBLE_DEVICES=0 bash ./31-test-wait-k.sh
```

## (Not avaiable now) MMA-IL (Monotonic Multihead Attention with Infinite Lookback) model
- Set variable `CUDA_VISIBLE_DEVICES`.
- You may use multiple GPUs, but the batch size becomes larger accordingly.
### Training
```shell
$ env CUDA_VISIBLE_DEVICES=0 bash ./40-train-st-mma-il.sh
```