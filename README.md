# InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE

<p align="left">
  <a href='https://arxiv.org/abs/2511.13488'>
    <img src='https://img.shields.io/badge/Arxiv-2511.13488-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2511.13488'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href=''>
  <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'></a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=lighten001.InterMoE&left_color=gray&right_color=orange">
  </a>
</p>

This repository contains the official implementation for the paper: [InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE (AAAI-26)](https://arxiv.org/abs/2511.13488).


## TODO

- [x] Evaluation codes.
- [x] Inference codes.
- [x] Checkpoints on InterHuman datasets.
- [ ] Training codes.
- [ ] Visualization codes.
- [ ] Checkpoints on Inter-X datasets.

## Getting started

This code was tested on `Ubuntu 20.04 LTS` and requires:

* Python 3.10
* conda3 or miniconda3
* CUDA capable GPU

### 1. Setup environment

```shell
conda create -n intermoe python=3.10
conda activate intermoe
pip install -r requirements.txt
```

### 2. Get datasets and some miscellaneous.


1. InterHuman: Download the InterHuman dataset from [their official webpage](https://tr3e.github.io/intergen-page/), and put them into ``data/InterHuman``. Then download the model for evaluation from [this script](https://github.com/tr3e/InterGen/blob/master/prepare/download_evaluation_model.sh).


2. Inter-X: Download the Inter-X dataset from [their official repo](https://github.com/liangxuy/Inter-X?tab=readme-ov-file#dataset-download). And put them into ``data/InterX``.

And finally the data structure should be like:

```
data/InterHuman/
    annots/
    motions/
    motions_processed/
    split/
    ...

data/InterX/
    h5/
    processed/
    splits/
    text2motion/
    ...

eval_model/
  ...
```


## Inference

### 1. Download the checkpoint

Download checkpoints from [Google drive](https://drive.google.com/file/d/1lsa2ifsicd-dW013IYXrtLRr6qcucmzy/view?usp=sharing), unzip the compressed file and put them under the ``checkpoints`` folder.

For example, it will be like:
```
checkpoints/
  intermoe-interhuman
    model/
      ...
    config.yaml
```

### 2. Modify the input file ``./prompts.txt`` like:

```
The two are blaming each other and having an intense argument.
Two fencers engage in a thrilling duel, their sabres clashing and sparking as they strive for victory.
Two individuals are practicing tai chi.
Two people bow to each other.
In an intense boxing match, one is continuously punching while the other is defending and counterattacking.
...
```

### 3. Run

Modify config files (``model.vae_ckpt`` and ``model.CHECKPOINT``) in ``CFG_PATH`` and run:

```shell
python infer.py --cfg ${CFG_PATH}

# for example, the CFG_PATH can be checkpoints/intermoe-interhuman/config.yaml.
```

The results will be in the ``GENERAL.CHECKPOINT/GENERAL.EXP_NAME/results`` sub-folder.


## Train


Coming soon.


## Evaluation


Modify config files (``model.vae_ckpt`` and ``model.CHECKPOINT``) in ``CFG_PATH`` and run:

```shell
# for interhuman
python eval_interhuman.py --cfg ${CFG_PATH}

# for interx
python eval_interx.py --cfg ${CFG_PATH}
```


## Acknowledgement

We appreciate the open source of the following projects:

[InterGen](https://github.com/tr3e/InterGen),
[Inter-X](https://github.com/liangxuy/Inter-X),
[salad](https://github.com/seokhyeonhong/salad),
[motion-latent-diffusion](https://github.com/ChenFengYe/motion-latent-diffusion),
[motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model), etc.



## Citation

If you find our work useful in your research, please consider citing:

```
@article{}
```

## Licenses

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
