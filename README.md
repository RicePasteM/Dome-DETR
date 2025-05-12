<h1 align="center">Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection</h1>

<p align="center">
    <a href="https://github.com/RicePasteM/Dome-DETR/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://github.com/RicePasteM/Dome-DETR/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/RicePasteM/Dome-DETR">
    </a>
    <a href="https://github.com/RicePasteM/Dome-DETR/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/RicePasteM/Dome-DETR?color=olive">
    </a>
    <a href="https://arxiv.org/abs/2505.05741">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.05741-red">
    </a>
</p>

<p align="center">
    This is the official implementation of the paper:
    <br>
    <a href="https://arxiv.org/abs/2505.05741">Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection</a>
</p>

![method](./static/method.png)

## Updates

[2025/xx/xx] We released the code of Dome-DETR.

### Setup

```shell
conda create -n dome python=3.11.9
conda activate dome
pip install -r requirements.txt
```

## Eval
```sh
# Change the parameters in dist_test.sh according to your need.
bash dist_test.sh
```

## Train
```sh
# Change the parameters in dist_train.sh according to your need.
bash dist_train.sh
```

## Citation
```bibtex
@misc{2505.05741,
  Author = {Zhangchi Hu and Peixi Wu and Jie Chen and Huyue Zhu and Yijun Wang and Yansong Peng and Hebei Li and Xiaoyan Sun},
  Title = {Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection},
  Year = {2025},
  Eprint = {arXiv:2505.05741},
}