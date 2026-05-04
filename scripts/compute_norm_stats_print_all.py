"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:

    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # 1) raw dataset
    raw_dataset = _data_loader.wocreate_torch_dataset(data_config, action_horizon, model_config)
    raw0 = raw_dataset[0]
    print("RAW keys:", raw0.keys())
    print("RAW actions dtype/shape:", np.asarray(raw0["actions"]).dtype, np.asarray(raw0["actions"]).shape)
    print("RAW actions[0]:", np.asarray(raw0["actions"])[0])

    # 2) transforms list
    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        RemoveStrings(),
    ]
    print("TRANSFORMS:", [type(t).__name__ for t in transforms])

    # 3) step-by-step apply on RAW sample
    x = raw0
    for i, t in enumerate(transforms):
        before = x.get("actions", None)
        x2 = t(x)

        after = x2.get("actions", None)
        if after is not None:
            a0 = np.asarray(after)[0]
            if before is not None:
                b0 = np.asarray(before)[0]
                diff0 = a0 - b0
                print(f"[{i:02d}] {type(t).__name__}: actions[0]={a0}  diff={diff0}")
            else:
                print(f"[{i:02d}] {type(t).__name__}: actions[0]={a0}  (actions appeared)")

        x = x2

    # 4) official transformed dataset (for verification)
    trans_dataset = _data_loader.TransformedDataset(raw_dataset, transforms)
    trans0 = trans_dataset[0]
    print("TRANS keys:", trans0.keys())
    print("TRANS actions dtype/shape:", np.asarray(trans0["actions"]).dtype, np.asarray(trans0["actions"]).shape)
    print("TRANS actions[0]:", np.asarray(trans0["actions"])[0])


    # dataloader
    if max_frames is not None and max_frames < len(trans_dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(trans_dataset) // batch_size
        shuffle = False

    data_loader = _data_loader.TorchDataLoader(
        trans_dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches





def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)

    data_config = config.data.create(config.assets_dirs, config.model)
    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
    )
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    # Print ALL inputs that are fed into normalize.RunningStats().
    # WARNING: this can produce a *lot* of output for large datasets.
    np.set_printoptions(threshold=np.inf, linewidth=200)

    for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats")):
        for key in keys:
            arr = np.asarray(batch[key])

            # Print each element along the batch dimension (i.e., per-sample in the batch).
            # If you truly want per-scalar printing, change this loop to iterate over arr.flat instead.
            # if arr.ndim == 0:
            #     print(f"[batch={batch_idx} key={key}] {arr}")
            # else:
            #     for item_idx, item in enumerate(arr):
            #         print(f"[batch={batch_idx} item={item_idx} key={key}] {item}")

            stats[key].update(arr)

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    print(f"norm_stats: {norm_stats}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)