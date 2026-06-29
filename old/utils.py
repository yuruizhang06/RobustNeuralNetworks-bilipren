import jax
import jax.numpy as jnp
import numpy as np
from collections import defaultdict

def l2_norm(x, eps=jnp.finfo(jnp.float32).eps, **kwargs):
    """Compute l2 norm of a vector/matrix with JAX.
    This is safe for backpropagation, unlike `jnp.linalg.norm`."""
    return jnp.sqrt(jnp.maximum(jnp.sum(x**2, **kwargs), eps))

def cayley(W):
    # W in shape n x 2n (m=2n)
    # W = [G H]
    m, n = W.shape 
    if n > m:
       return cayley(W.T).T
    
    G, H = W[:n, :], W[n:, :]

    # Z = GT-G + HTH -------- Eq6
    Z = (G - G.T) + (H.T @ H)
    I = jnp.eye(n)
    Zi = jnp.linalg.inv(I+Z)

    # (I+Z)(I-z)-1    -2V(I-Z)-1
    return jnp.concatenate([Zi @ (I-Z), -2 * H @ Zi], axis=0)

def identity_init():
    """Initialize a weight as the identity matrix.
    
    Assumes that shape is a tuple (n,n), only uses first element.
    """
    def init(key, shape, dtype):
        return jnp.identity(shape[0], dtype)
    return init

def _prepare_indices(n, key, shuffle):
    if shuffle:
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, n)
        return key, indices
    return key, jnp.arange(n)


def _iter_batches(indices, batch_size, drop_last):
    n = indices.shape[0]
    end = n - (n % batch_size) if drop_last else n
    for i in range(0, end, batch_size):
        yield indices[i:i + batch_size]

def _is_ragged(data):
    if isinstance(data, (list, tuple)):
        return True
    return isinstance(data, np.ndarray) and data.dtype == object


def _dataset_len(data):
    return len(data) if isinstance(data, (list, tuple)) else data.shape[0]


def _batch_select(data, batch_indices, padding_value=0):
    if _is_ragged(data):
        batch = [data[int(i)] for i in batch_indices]
        return pad_sequences_in_batch(batch, padding_value=padding_value)
    return data[batch_indices]


def data_generator(
    training_in,
    training_out,
    batch_size,
    epochs,
    key,
    shuffle=True,
    drop_last=False,
    bucket_boundaries=None,
    padding_value=0,
    x0=None,
):
    """
    Unified data generator for fixed-length or ragged sequences.
    - training_out can be None for single-input datasets.
    - x0 can be provided for partial PL training.
    - bucket_boundaries enables length-based bucketing for ragged inputs.
    """
    n = _dataset_len(training_in)
    use_bucket = bucket_boundaries is not None or _is_ragged(training_in)

    if not use_bucket:
        for epoch in range(epochs):
            key, indices = _prepare_indices(n, key, shuffle)
            for batch_idx, batch_indices in enumerate(_iter_batches(indices, batch_size, drop_last)):
                batch_in = training_in[batch_indices]
                batch_out = training_out[batch_indices] if training_out is not None else None
                if x0 is None:
                    if batch_out is None:
                        yield epoch, batch_idx, batch_in
                    else:
                        yield epoch, batch_idx, batch_in, batch_out
                else:
                    yield epoch, batch_idx, batch_in, batch_out, x0[batch_indices]
        return

    grouped_indices = defaultdict(list)
    if bucket_boundaries is None:
        for i, seq in enumerate(training_in):
            grouped_indices[len(seq)].append(i)
    else:
        boundaries = np.array(bucket_boundaries)
        for i, seq in enumerate(training_in):
            bucket_idx = np.searchsorted(boundaries, len(seq))
            grouped_indices[bucket_idx].append(i)

    bucket_indices = {k: jnp.array(v) for k, v in grouped_indices.items()}

    for epoch in range(epochs):
        all_batches = []
        for _, idxs in bucket_indices.items():
            key, bucket_perm = _prepare_indices(idxs.shape[0], key, shuffle)
            bucket_idx = idxs[bucket_perm]
            for batch_indices in _iter_batches(bucket_idx, batch_size, drop_last):
                all_batches.append(batch_indices)

        n_batches = len(all_batches)
        if shuffle:
            key, subkey = jax.random.split(key)
            order = jax.random.permutation(subkey, n_batches)
        else:
            order = jnp.arange(n_batches)

        for batch_idx, order_idx in enumerate(order):
            batch_indices = all_batches[int(order_idx)]
            batch_in = _batch_select(training_in, batch_indices, padding_value=padding_value)
            batch_out = None
            if training_out is not None:
                batch_out = _batch_select(training_out, batch_indices, padding_value=padding_value)
            if x0 is None:
                if batch_out is None:
                    yield epoch, batch_idx, batch_in
                else:
                    yield epoch, batch_idx, batch_in, batch_out
            else:
                yield epoch, batch_idx, batch_in, batch_out, _batch_select(x0, batch_indices, padding_value=padding_value)



def pad_sequences_in_batch(batch_sequences, padding_value=0):
    max_len = max(len(seq) for seq in batch_sequences)
    feature_dim = batch_sequences[0].shape[1] if batch_sequences[0].ndim > 1 else 1
    padded_batch = np.full((len(batch_sequences), max_len, feature_dim), padding_value, dtype=np.float32)
    for i, seq in enumerate(batch_sequences):
        seq_len = len(seq)
        padded_batch[i, :seq_len] = seq.reshape(seq_len, feature_dim)  
    return jnp.array(padded_batch)


def l2_norm_metric(y_true, y_pred, time_axis=None, reduce="mean", eps=1e-12):
    """
    General L2 norm metric for 2D or 3D arrays.
    - 2D: (batch, dim) or (n, dim) -> L2 over dim.
    - 3D: (time, batch, dim) -> L2 over dim, then sum over time.
    Use time_axis to override default if needed.
    """
    err = y_true - y_pred
    per_step = jnp.sum(jnp.square(err), axis=-1)

    if time_axis is None:
        time_axis = 0 if per_step.ndim >= 2 else None

    if time_axis is None:
        per_traj = jnp.sqrt(jnp.maximum(per_step, eps))
    else:
        per_traj = jnp.sqrt(jnp.maximum(jnp.sum(per_step, axis=time_axis), eps))

    if reduce == "none":
        return per_traj
    if reduce == "sum":
        return jnp.sum(per_traj)
    if reduce == "mean":
        return jnp.mean(per_traj)
    raise ValueError(f"Unknown reduce='{reduce}'. Use 'mean', 'sum', or 'none'.")


def l2_norm_loss(y_true, y_pred):
    return l2_norm_metric(y_true, y_pred, time_axis=0, reduce="mean")


def l2_norm_perbatch(y_true, y_pred):
    return l2_norm_metric(y_true, y_pred, time_axis=0, reduce="none")


def l2_norm_loss_2d(y_true, y_pred):
    return l2_norm_metric(y_true, y_pred, time_axis=None, reduce="mean")

def compute_lipschitz_constants(x_samples, y_samples, num_samples, rng):
    max_lipschitz = 0
    min_inverse_lipschitz = jnp.inf
    
    for i in range(num_samples):
        rng, subkey = jax.random.split(rng)
        indices = jax.random.choice(subkey, x_samples.shape[0], shape=(2,), replace=False)
        delta_x = x_samples[indices[0]] - x_samples[indices[1]]
        delta_y = y_samples[indices[0]] - y_samples[indices[1]]
        norm_x = jnp.linalg.norm(delta_x)
        norm_y = jnp.linalg.norm(delta_y)
        lipschitz_const = norm_y / norm_x
        
        max_lipschitz = max(max_lipschitz, lipschitz_const)
        min_inverse_lipschitz = min(min_inverse_lipschitz, lipschitz_const)
    
    return max_lipschitz, min_inverse_lipschitz

def normalize_to_unit(z, z_min, z_max, eps=1e-12):
    """
    Normalize z to [-1, 1]
    """
    return 2.0 * (z - z_min) / (z_max - z_min + eps) - 1.0


def preprocess_data(
    inputs,
    outputs,
    input_min=None,
    input_max=None,
    output_min=None,
    output_max=None,
    clip_inputs=None,
    clip_outputs=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle=True,
    rng=None,
    time_major=False,
):
    """
    Preprocess data with normalization, clipping, split, and dimension alignment.

    - Inputs/outputs are expected as (N, T, D) or (N, D). Outputs can be None.
    - Normalization is min-max to [-1, 1] per provided ranges.
    - Clipping uses provided (min, max) tuples before normalization.
    - If time_major=True, returns (T, N, D) for sequence data.
    """
    if outputs is not None and inputs.shape[0] != outputs.shape[0]:
        raise ValueError("inputs and outputs must have the same first dimension")

    if clip_inputs is not None:
        inputs = jnp.clip(inputs, clip_inputs[0], clip_inputs[1])
    if outputs is not None and clip_outputs is not None:
        outputs = jnp.clip(outputs, clip_outputs[0], clip_outputs[1])

    if input_min is not None and input_max is not None:
        inputs = normalize_to_unit(inputs, input_min, input_max)
    if outputs is not None and output_min is not None and output_max is not None:
        outputs = normalize_to_unit(outputs, output_min, output_max)

    n = inputs.shape[0]
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    if shuffle:
        if rng is None:
            rng = jax.random.key(0)
        perm = jax.random.permutation(rng, n)
    else:
        perm = jnp.arange(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    def _to_time_major(arr):
        if arr is None:
            return None
        if arr.ndim < 3:
            return arr
        return jnp.transpose(arr, axes=(1, 0, 2))

    def _slice(arr, idx):
        sliced = arr[idx] if arr is not None else None
        if time_major:
            return _to_time_major(sliced)
        return sliced

    result = {
        "train": (_slice(inputs, idx_train), _slice(outputs, idx_train)),
        "val": (_slice(inputs, idx_val), _slice(outputs, idx_val)),
        "test": (_slice(inputs, idx_test), _slice(outputs, idx_test)),
        "indices": (idx_train, idx_val, idx_test),
        "norm": {
            "input_min": input_min,
            "input_max": input_max,
            "output_min": output_min,
            "output_max": output_max,
        },
        "sizes": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
    }
    return result