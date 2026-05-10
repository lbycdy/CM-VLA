# CM-VLA
CM-VLA is a camera-space-guided mixture-of-experts policy for few-shot cross-embodiment transfer.

## Requirements
To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 9 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 23 GB         | RTX 4090           |
| Fine-Tuning (Full) | > 75 GB         | A800 (80GB)        |

The repo has been tested with Ubuntu 20.04, we do not currently support other operating systems.

## Installation

You can refer to the [HTTS](https://github.com/Physical-Intelligence/openpi) installation environment. When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.
