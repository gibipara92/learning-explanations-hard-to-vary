### Learning explanations that are hard to vary
This repo contains the code to implement the methods from the paper _Learning explanations that are hard to vary_ [arxiv.org/abs/2009.00329](https://arxiv.org/abs/2009.00329).

#### Instructions
To run the baseline (standard SGD), use `method='and_mask'` and `agreement_threshold=0.`.

There are two examples:
### Synthetic dataset

```
python -m and_mask.run_synthetic \
        --method=and_mask \
        --agreement_threshold=1. \
        --n_train_envs=16 \
        --n_agreement_envs=16 \
        --batch_size=256 \
        --n_dims=16 \
        --scale_grad_inverse_sparsity=1 \
        --use_cuda=1 \
        --n_hidden_units=256
```

### CIFAR-10

```
python -m and_mask.run_cifar \
        --random_labels_fraction 1.0 \
        --agreement_threshold 0.2 \
        --method and_mask \
        --epochs 80 \
        --weight_decay 1e-06 \
        --scale_grad_inverse_sparsity 1 \
        --init_lr 0.0005 \
        --weight_decay_order before \
        --output_dir /tmp/
```

#### Bibtex

```
@misc{parascandolo2020learning,
      title={Learning explanations that are hard to vary}, 
      author={Giambattista Parascandolo and Alexander Neitz and Antonio Orvieto and Luigi Gresele and Bernhard Schölkopf},
      year={2020},
      eprint={2009.00329},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
