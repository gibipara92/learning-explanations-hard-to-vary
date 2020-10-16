import numpy as np

def get_spirals_dataset(n_examples, n_rotations, env, n_envs,
                        n_dims_signatures,
                        seed=None):
    """
    env must either be "test" or an int between 0 and n_envs-1
    n_dims_signatures: how many dimensions for the signatures (spirals are always 2)
    seed: seed for numpy
    """
    assert env == 'test' or 0 <= int(env) < n_envs

    # Generate fixed dictionary of signatures
    rng = np.random.RandomState(seed)

    signatures_matrix = rng.randn(n_envs, n_dims_signatures)

    radii = rng.uniform(0.08, 1, n_examples)
    angles = 2 * n_rotations * np.pi * radii

    labels = rng.randint(0, 2, n_examples)
    angles = angles + np.pi * labels

    radii += rng.uniform(-0.02, 0.02, n_examples)
    xs = np.cos(angles) * radii
    ys = np.sin(angles) * radii

    if env == 'test':
        signatures = rng.randn(n_examples, n_dims_signatures)
    else:
        env = int(env)
        signatures_labels = np.array(labels * 2 - 1).reshape(1, -1)
        signatures = signatures_matrix[env] * signatures_labels.T

    signatures = np.stack(signatures)
    mechanisms = np.stack((xs, ys), axis=1)
    mechanisms /= mechanisms.std(axis=0)  # make approx unit variance (signatures already are)
    inputs = np.hstack((mechanisms, signatures))

    return inputs.astype(np.float32), labels.astype(np.float32)