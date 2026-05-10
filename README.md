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

You can refer to the [OpenPi](https://github.com/Physical-Intelligence/openpi) installation environment. When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:lbycdy/CM-VLA.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | 
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | 
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | 

### Fine-Tuned Models
You can fine-tune the "CM-expert" checkpoints for various robot platforms and tasks. These models can be fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case    | Description                                                                                                                                                                                              | 
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\pi_{0.5}$-LIBERO      | Fine-Tuning / Inference   | CM-VLA model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) |
| $\pi_{0.5}$-ME-LIBERO          | Fine-Tuning / Inference | CM-VLA model fine-tuned for the [ME-LIBERO](https://huggingface.co/datasets/lbycdy/ME-LIBERO)  | 

## Fine-Tuning Base Models on ME-LIBERO or Your Own Data

You can fine-tune the CM-VLA model on the [ME-LIBERO dataset](https://huggingface.co/datasets/lbycdy/ME-LIBERO) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert ME-LIBERO or your data to a LeRobot dataset

We provide a minimal example script for converting ME-LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw ME-LIBERO dataset from [here]([https://huggingface.co/datasets/openvla/modified_libero_rlds](https://huggingface.co/datasets/lbycdy/ME-LIBERO)), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```
### 2. Defining training configs and running training

To fine-tune a base model on ME-LIBERO or your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for ME-LIBERO below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

We provide example fine-tuning configs for [CM-VLA](src/openpi/training/config.py) on ME-LIBERO data.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name cmvla_libero
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py cmvla_libero --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a ME-LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 60,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=cmvla_libero --policy.dir=checkpoints/cmvla_libero/my_experiment/60000
```
