import torch
from torch.utils.data import TensorDataset

from and_mask.datasets.synthetic.synthetic_data_gen import get_spirals_dataset

def make_dataloader(n_examples, env, n_envs, n_revolutions, n_dims,
                    batch_size,
                    use_cuda,
                    flip_first_signature=False,
                    seed=None):

    inputs, labels = get_spirals_dataset(n_examples,
                                         n_rotations=n_revolutions,
                                         env=env,
                                         n_envs=n_envs,
                                         n_dims_signatures=n_dims,
                                         seed=seed
                                         )
    if flip_first_signature:
        inputs[:1, 2:] = -inputs[:1, 2:]

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    data_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.tensor(inputs), torch.tensor(labels)),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    return data_loader

