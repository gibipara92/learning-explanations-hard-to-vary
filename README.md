### Learning explanations that are hard to vary
This repo contains the code to implement the methods from the paper _Learning explanations that are hard to vary_ [arxiv.org/abs/2009.00329](https://arxiv.org/abs/2009.00329).

#### Interested in trying it on a new dataset?
In our experience, here are the very important hyperparameters to tune, that we would include in a wide hyperparameter search:

- **higher lr**: In the hyper parameter search we usually set it on a log_range(1e-4, 1e-0)
- **weight decay**:
	- Potentially much higher than usual, we usually search on a log_range(1e-5, 1e-0)
	- _(already default)_ It's applied after the mask (so it affects the weights even for masked features)
- **inverse scaling**: Rescales the remaining gradients by the ratio of entries that survived the mask in each layer. _(This is a pretty extreme re-scaling, we haven’t tried any other so far)_. We add `scale_grad_inverse_sparsity` as a boolean hyperparam in the search.
- **geom mean**: In some cases (e.g. if there is some noise _and_ few environments, as it's the case for the notebook) the and_mask approximation is worse, and it’s best to go for the geom mean (the downside is that gradients get even smaller). We also just set this as an option in the hyperaparameter search (`method`).
- **optimizer**: Adam or SGD. Adam rescales gradients, so the two can behave quite differently.
- **agreement_threshold**: 1 might work best in some synthetic environments, but might be too strict for real life environments. Definitely search this too.

Let us know what ends up working best, so hopefully over time we can make this list and ranges shorter =)

#### Instructions
To run the baseline (standard SGD), use `method='and_mask'` and `agreement_threshold=0.`

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

### A simple linear regression example

See the notebook folder.

#### BibTeX

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
