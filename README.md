# MemER

Code for the paper ["MemER: Scaling Up Memory for Robot Control via Experience Retrieval"](https://arxiv.org/abs/2510.20328). This repo contains the high-level training data based on the Qwen3-VL finetune format.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Data Prep Requirements
Once uv is installed, run the following to set up the data conversion environment:

```bash
uv sync
```

### Eval / Deployment Requirements

Install a CUDA-enabled `torch` build that matches your machine before running the command below if it is not already present.

```bash
uv sync --extra eval
```

Optionally install `ffmpeg` if you want rendered rollout videos.

## High-Level Policy

### Data Prep

#### Prepare a LeRobot Dataset

[`generate_sft_data.py`](scripts/generate_sft_data.py) expects a LeRobot v3 dataset with task and subtask metadata. If you need to create or edit subtask annotations, use the [LeRobot subtask guide](https://huggingface.co/docs/lerobot/dataset_subtask).

We provide an example using the datasets for the "Dust & Replace" task from our paper:

- **Training**: [ajaysri/dusting_train_50_v3_subtasks_rgb](https://huggingface.co/datasets/ajaysri/dusting_train_50_v3_subtasks_rgb)
- **Evaluation**: [ajaysri/dusting_test_10_v3_subtasks_rgb](https://huggingface.co/datasets/ajaysri/dusting_test_10_v3_subtasks_rgb)

For the dusting datasets above, the canonical [Franka Emika Panda](https://droid-dataset.github.io/droid/) stacked-camera layout lives in [`configs/robots/panda_stacked_rgb_camera_layout.json`](configs/robots/panda_stacked_rgb_camera_layout.json). If multiple tasks share the same camera setup, reuse the same robot-level camera-layout config across export, eval, and deployment so the image stack order stays identical across all three stages.

#### Export Qwen SFT Data

[`generate_sft_data.py`](scripts/generate_sft_data.py) reads the LeRobot dataset and writes:

```text
output_dir/
  train.json
  media/
    episode_000000/
      frame_000000.jpg
      frame_000001.jpg
      ...
```

`train.json` uses Qwen-style `image` + `conversations` records. Image paths are relative to `output_dir`.

##### Example Usage

```bash
python3 scripts/generate_sft_data.py \
  --repo_id ajaysri/dusting_train_50_v3_subtasks_rgb \
  --output_dir ./release/dusting_train \
  --camera_layout_config configs/robots/panda_stacked_rgb_camera_layout.json \
  --keyframe_rule_file configs/tasks_from_paper/dusting_keyframe_rules.json \
  --high_level_instruction "What subtask should the robot execute to remove the items from the shelves, dust the shelves, and place the items back on the shelves?" \
  --frame_subsample 5 \
  --recent_frames_length 8 \
  --keyframes_length 8 \
  --prediction_horizon 2 \
  --num_workers 4
```

##### Flags

- `--lerobot_path`: Local LeRobot dataset root. If provided, data is loaded directly from this directory and the directory name is used as the LeRobot dataset identifier. If omitted, LeRobot downloads and caches the dataset from Hugging Face Hub using `--repo_id`.
- `--repo_id`: Dataset identifier for LeRobot (e.g. `ajaysri/dusting_train_50_v3_subtasks_rgb`). Required when `--lerobot_path` is not set.
- `--camera_layout_config`: Path to the JSON config that defines the ordered `camera_keys` and per-view image size. 
- `--keyframe_rule_file`: Path to the JSON config that decides what keyframes should be selected based on the wording of the subtask (e.g. you can define a rule that select all subtasks with the words "look"). For some examples, see [`configs/tasks_from_paper/`](configs/tasks_from_paper) (e.g. [`dusting_keyframe_rules.json`](configs/tasks_from_paper/dusting_keyframe_rules.json), [`object_search_keyframe_rules.json`](configs/tasks_from_paper/object_search_keyframe_rules.json), [`counting_keyframe_rules.json`](configs/tasks_from_paper/counting_keyframe_rules.json)).
- `--frame_subsample`: Subsamples the video data from the LeRobot dataset to every Nth frame.
- `--recent_frames_length`: The number of frames in the high-level policy's rolling recent window after subsampling.
- `--keyframes_length`: The maximum number of memory keyframes added to the high-level policy's context (context is a FIFO queue) after subsampling.
- `--prediction_horizon`: Shifts the target label forward by `prediction_horizon * frame_subsample` frames.

**Camera views and image sizing.** Multiple camera views are stacked vertically into a single image. You specify which camera keys to use and the size of each individual view (`view_width` x `view_height`) in a layout config (see [`panda_stacked_rgb_camera_layout.json`](configs/robots/panda_stacked_rgb_camera_layout.json) for an example), and pass it via `--camera_layout_config`. The final stacked image width equals `view_width` and its height equals `view_height * number_of_cameras`.

### Training

For the full Qwen3-VL finetuning recipe, see [TRAINING_QWEN3VL.md](TRAINING_QWEN3VL.md). That document includes:

- the Qwen dataset-registration step
- the 1500-step reference training command
- the key 4B dusting training settings

The public MemER flow remains:

1. export LeRobot data with `generate_sft_data.py`
2. register the exported `train.json` + `media/` directory in Qwen
3. finetune Qwen3-VL on that export

### Pretrained Checkpoints

A finetuned checkpoint for the dusting task is available on Hugging Face:

| Task | Base Model | Checkpoint | Steps |
|------|-----------|-----------|-------|
| Dust & Replace | `Qwen/Qwen3-VL-4B-Instruct` | [`ajaysri/memer-dusting-qwen3vl-4b-step-1500`](https://huggingface.co/ajaysri/memer-dusting-qwen3vl-4b-step-1500) | 1500 |

**Download locally:**

```bash
huggingface-cli download ajaysri/memer-dusting-qwen3vl-4b-step-1500 \
  --local-dir ./ckpts/checkpoint-1500
```

**Use in code** (Hub ID or local path both work for `model_path`):

```python
policy = MemERDeploymentPolicy.from_qwen_checkpoint(
    model_path="ajaysri/memer-dusting-qwen3vl-4b-step-1500",
    ...
)
```

### Deployment

#### Online Deployment API

Use [`MemERDeploymentPolicy`](memer_eval/deploy.py) when you want a stateful drop-in wrapper for deployment code. It owns the recent-frame buffer, episodic memory clustering, prompt construction, and model calls.

Relevant API surface:

- [`MemERDeploymentPolicy.from_qwen_checkpoint()`](memer_eval/deploy.py): load the finetuned Qwen checkpoint and create a stateful deployment wrapper.
- [`MemERDeploymentPolicy.reset()`](memer_eval/deploy.py): start a new task or episode and clear recent context and memory.
- [`MemERDeploymentPolicy.step()`](memer_eval/deploy.py): push one or more observations and get the next predicted subtask. Accepts a single observation or a chunk — inference runs once at the final timestep either way.

Internally, the deployment wrapper uses:

- [`build_human_prompt()`](memer_eval/contract.py) for the release prompt contract
- [`EpisodicMemory`](memer_eval/memory.py) for candidate clustering and visible memory keyframes
- [`QwenStructuredPredictor`](memer_eval/inference.py) for model inference and JSON parsing

`step()` is the single entry point for all observation patterns. Pass one observation per call in a live control loop, or pass a chunk of observations to buffer them all and run inference once at the end. The policy stays stateful across calls either way. Each observation can be a PIL image, numpy array, torch tensor, or the dict-style image object returned by LeRobot.

```python
from memer_eval import DeploymentConfig, MemERDeploymentPolicy

policy = MemERDeploymentPolicy.from_qwen_checkpoint(
    model_path="./ckpts/checkpoint-1500",
    processor_path="./ckpts/checkpoint-1500",
    instruction="What subtask should the robot execute to remove the items from the shelves, dust the shelves, and place the items back on the shelves?",
    device="cuda",
    dtype="bfloat16",
    attn_implementation="sdpa",
    max_new_tokens=64,
    config=DeploymentConfig(
        camera_layout_config="configs/robots/panda_stacked_rgb_camera_layout.json",
        frame_subsample=5,
        recent_frames_length=8,
        memory_length=8,
        merge_distance=5,  # in subsampled timesteps
    ),
)
```

Pass observations to `step()` — one at a time or as a chunk:

```python
# Single-camera: pass the raw image directly
result = policy.step(wrist_image)

# Multi-camera: pass a dict keyed by camera name
result = policy.step({
    "observation.images.wrist_left": wrist_image,
    "observation.images.exterior_1_left": exterior_image,
})

# Chunk of observations: buffer all frames, infer once at the end
result = policy.step(
    {"observation.images.wrist_left": wrist_t0, "observation.images.exterior_1_left": ext_t0},
    {"observation.images.wrist_left": wrist_t1, "observation.images.exterior_1_left": ext_t1},
    {"observation.images.wrist_left": wrist_t2, "observation.images.exterior_1_left": ext_t2},
)

# Or unpack an existing list
result = policy.step(*observation_history)

print(result.current_subtask)
print(result.memory_indices_before, result.context_indices)
```

Image values can be PIL images, numpy arrays, torch tensors, or the dict-style image objects returned by LeRobot.

#### Offline Rollout Eval

[`eval_subtask_rollout.py`](scripts/eval_subtask_rollout.py) replays LeRobot episodes sequentially, rebuilds the rollout prompt from the training contract, updates memory from model-predicted keyframes with 1D clustering, and reports subtask accuracy.

##### Data And Checkpoint Contract

- `--model-path` must point to a local Qwen3-VL checkpoint directory.
- `--lerobot-path` must point to the LeRobot dataset root and include `meta/info.json`, `meta/tasks.parquet`, and `meta/subtasks.parquet`.

Use the [pretrained checkpoint](#pretrained-checkpoints) to get started quickly, or train your own with the recipe in [TRAINING_QWEN3VL.md](TRAINING_QWEN3VL.md).

##### Example

```bash
python3 scripts/eval_subtask_rollout.py \
  --model-path ./ckpts/checkpoint-1500 \
  --processor-path /path/to/Qwen3-VL-4B-Instruct \
  --lerobot-path ./data/dusting_test_10_v3_subtasks_rgb \
  --repo-id ajaysri/dusting_test_10_v3_subtasks_rgb \
  --output-dir ./eval_outputs/dusting_test_rollout \
  --high-level-instruction "What subtask should the robot execute to remove the items from the shelves, dust the shelves, and place the items back on the shelves?" \
  --camera-layout-config configs/robots/panda_stacked_rgb_camera_layout.json \
  --frame-subsample 5 \
  --recent-frames-length 8 \
  --memory-length 8 \
  --prediction-horizon 2 \
  --merge-distance 5 \
  --attn-implementation sdpa \
  --save-raw-responses \
  --max-episodes 1  # remove this to eval on all episodes
```

The evaluator writes:

- `summary.json`: overall metrics and runtime/config metadata
- `episodes.json`: per-episode accuracy
- `label_metrics.json`: per-subtask accuracy
- `predictions.jsonl`: one row per rollout timestep

If you omit `--camera-layout-config` and `--camera-keys`, the evaluator first looks for the repo's preferred two-camera pair and otherwise falls back to the first available camera. For portable scripts, prefer one shared camera-layout config.

#### Reference Result

- **Dataset:** [`ajaysri/dusting_test_10_v3_subtasks_rgb`](https://huggingface.co/datasets/ajaysri/dusting_test_10_v3_subtasks_rgb)
- **Model:** `Qwen/Qwen3-VL-4B-Instruct` finetuned on the 50-episode dusting train set
- **Accuracy:** `94.21%`
- **Number of Episodes**: `10`
- **Total Timesteps:** `6162`

This number uses the Panda camera-layout config above and the full offline rollout eval path in this repo. It is the current reference target for the public dusting release.

## Low-Level Policy

The low-level policy is decoupled from the MemER high-level policy choice. This section is about how to train and deploy a low-level policy that consumes subtask predictions from the high-level policy.

- We recommend starting from [openpi](https://github.com/Physical-Intelligence/openpi), which provides open-source VLA checkpoints and fine-tuning code.
- Replace the episode-level task instruction with the current subtask label for each training example so the low-level policy learns to execute subtasks rather than the full task description.
- The same pattern can be applied to other low-level policies such as GR00T, SmolVLA, or another policy of your choice.

## Deploying the High-Level and Low-Level Policies

- Run the high-level policy online to predict the current subtask from memory and recent observations.
- Condition the low-level policy on that predicted subtask instead of the original task description.
- Keep the robot observation and action interface unchanged; the integration point is the language command fed into the low-level policy.

The example below shows both policies running together in a control loop. See [Online Deployment API](#online-deployment-api) for details on `MemERDeploymentPolicy` setup and configuration.

```python
from memer_eval import DeploymentConfig, MemERDeploymentPolicy

high_level_policy = MemERDeploymentPolicy.from_qwen_checkpoint(
    model_path="./ckpts/checkpoint-1500",
    processor_path="./ckpts/checkpoint-1500",
    instruction=task_instruction,
    device="cuda",
    dtype="bfloat16",
    attn_implementation="sdpa",
    max_new_tokens=64,
    config=DeploymentConfig(
        camera_layout_config="configs/robots/panda_stacked_rgb_camera_layout.json",
        frame_subsample=5,
        recent_frames_length=8,
        memory_length=8,
        merge_distance=5,
    ),
)

low_level_policy = load_low_level_policy(...)

high_level_policy.reset(instruction=task_instruction)
obs = env.reset()

while True:
    result = high_level_policy.step(
        {
            "observation.images.wrist_left": obs["observation.images.wrist_left"],
            "observation.images.exterior_1_left": obs["observation.images.exterior_1_left"],
        }
    )

    action = low_level_policy.act(
        observation=obs,
        language_command=result.current_subtask,
    )

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Citation

If this codebase helps your research, please cite:

```bibtex
@inproceedings{sridhar2026memer,
  title={MemER: Scaling Up Memory for Robot Control via Experience Retrieval},
  author={Ajay Sridhar and Jennifer Pan and Satvik Sharma and Chelsea Finn},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2510.20328},
}
```
