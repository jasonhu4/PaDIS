## Learning Image Priors through Patch-based Diffusion Models for Solving Inverse Problems<br><sub>Official PyTorch implementation</sub>

![Teaser image](./docs/reconstruction.png)

**Learning Image Priors through Patch-based Diffusion Models for Solving Inverse Problems**<br>
Jason Hu, Bowen Song, Xiaojian Xu, Liyue Shen, Jeffrey A. Fessler
<br>https://arxiv.org/abs/2406.02462 <br>

Abstract: *Diffusion models can learn strong image priors from underlying data distribution and use them to solve inverse problems, but the training process is computationally expensive and requires lots of data. Such bottlenecks prevent most existing works from being feasible for high-dimensional and high-resolution data such as 3D images. This paper proposes a method to learn an efficient data prior for the entire image by training diffusion models only on patches of images. Specifically, we propose a patch-based position-aware diffusion inverse solver, called PaDIS, where we obtain the score function of the whole image through scores of patches and their positional encoding and utilize this as the prior for solving inverse problems. First of all, we show that this diffusion model achieves an improved memory efficiency and data efficiency while still maintaining the capability to generate entire images via positional encoding. Additionally, the proposed PaDIS model is highly flexible and can be plugged in with different diffusion inverse solvers (DIS). We demonstrate that the proposed PaDIS approach enables solving various inverse problems in both natural and medical image domains, including CT reconstruction, deblurring, and superresolution, given only patch-based priors. Notably, PaDIS outperforms previous DIS methods trained on entire image priors in the case of limited training data, demonstrating the data efficiency of our proposed approach by learning patch-based prior.*


## Requirements
* Python libraries: See [environment.yml](./environment.yml) for exact library dependencies.
* Also see [odl_env.yml](.odlstuff/odl_env.yml) for help on installing ODL package for running CT experiments.

## Getting started

### Preparing datasets


### Train Patch Diffusion

You can train new models using `train.py`. For example:

```.bash
# Train DDPM++ model for CelebA-64x64 using 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/celeba-64x64.zip --cond=0 --arch=ddpmpp --batch=256 \
    --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0 --real_p=0.5

# Train ADM model with Latent Diffusion Encoder for LSUN-256x256 using 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/lsun-bedroom-256x256.zip --cond=0 --arch=adm --train_on_latents=1 \
    --duration=200 --batch-gpu=32 --batch=1024 --lr=1e-4 --ema=50 --dropout=0.10 --fp16=1 --ls=100 \
    --augment=0 --real_p=0.5

# Train ADM model with Latent Diffusion Encoder for ImageNet-256x256 using 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/imagenet-256x256.zip --cond=1 --arch=adm --train_on_latents=1 \
    --duration=2500 --batch-gpu=32 --batch=4096 --lr=1e-4 --ema=50 --dropout=0.10 --fp16=1 --ls=100 \
    --augment=0 --real_p=0.5 --tick=200
```

We follow the hyperparameter settings of EDM, and introduce two new parameters here:

- `--real_p`: the ratio of full size image used in the training.
- `--train_on_latents`: where to train on the Latent Diffusion latent space, instead of the pixel space. Note we trained our models on the latent space for 256x256 images.

### Inference Patch Diffusion

You can generate images using `generate.py`. For example:
```.bash
# For DDPM++ Architecture we use
torchrun --standalone --nproc_per_node=8 generate.py --steps=50 --resolution 64 --batch 64 --outdir=fid-tmp --seeds=0-49999 --subdirs --network=/path-to-the-pkl/

# For ADM Architecture we use
torchrun --standalone --nproc_per_node=8 generate.py --steps=256 --S_churn=40 --S_min=0.05 --S_max=50 --S_noise=1.003 --resolution 32 --on_latents=1 --batch 64 --outdir=fid-tmp --seeds=0-49999 --subdirs --network=/path-to-the-pkl/
```

The model checkpoints that we trained will be released soon.

### Calculating FID

To compute Fr&eacute;chet inception distance (FID) for a given model and sampler, first generate 50,000 random images and then compare them against the dataset reference statistics using `fid.py`:

```.bash
# Generate 50000 images and save them as fid-tmp/*/*.png
torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl


## Citation

```
@article{hu2024padis,
  title={Learning Image Priors through Patch-based Diffusion Models for Solving Inverse Problems},
  author={Hu, Jason and Song, Bowen and Xu, Xiaojian and Shen, Liyue and Fessler, Jeffrey A.},
  journal={arXiv preprint arXiv:2406.02462},
  year={2024}
}
```

## Acknowledgments

We thank the [Patch-Diffusion](https://github.com/Zhendong-Wang/Patch-Diffusion) authors for providing a great code base.
