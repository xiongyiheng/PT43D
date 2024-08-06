# PT43D: A Probabilistic Transformer for Generating 3D Shapes from Single Highly-Ambiguous RGB Images
[[`arXiv`](https://arxiv.org/abs/2405.11914)]
[[`BibTex`](#citation)]
[[`Video`](https://youtu.be/b_-U7dXalAs?si=twE6gemtEQ4bUJ6h)]

Code release for BMVC 2024 paper "PT43D: A Probabilistic Transformer for Generating 3D Shapes from Single Highly-Ambiguous RGB Images".

![1-teaser-v3-out-1](docs/teaser.png)
We propose a new approach for generating the probabilistic distribution of 3D shape reconstructions conditioned on a highly ambiguous RGB image, enabling multiple diverse sampled hypotheses during inference.

# Installation
Please setup the environment using `conda`:

```
conda env create -f pt43d.yaml
conda activate pt43d
```

# Preparing the Data
1. [ShapeNet](https://www.shapenet.org)

Coming soon.

2. [ScanNet](http://www.scan-net.org/)

Coming soon.

# Training
1. First train the `P-VQ-VAE` on `ShapeNet`:
```
./launchers/train_pvqvae_snet.sh
```

2. Then extract the code for each sample of ShapeNet (caching them for training the transformer):
```
./launchers/extract_pvqvae_snet.sh
```

3. Train the probabilistic transformer to learn the shape distribution conditioned on an RGB image:
```
./launchers/train_pt43d.sh
```
# <a name="citation"></a> Citation

If you find this code helpful, please consider citing:
```BibTeX
@misc{xiong2024pt43d,
      title={PT43D: A Probabilistic Transformer for Generating 3D Shapes from Single Highly-Ambiguous RGB Images}, 
      author={Yiheng Xiong and Angela Dai},
      year={2024},
      eprint={2405.11914},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement
This work was supported by the ERC Starting Grant SpatialSem (101076253). This code borrowed heavily from [AutoSDF](https://github.com/yccyenchicheng/AutoSDF). Thanks for the efforts for making their code available!
