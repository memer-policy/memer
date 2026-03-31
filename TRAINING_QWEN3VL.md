# Qwen3-VL Training

This document covers the reference Qwen3-VL finetuning setup used for the MemER high-level policy after you have already exported a LeRobot dataset into the Qwen release format with [`scripts/generate_sft_data.py`](scripts/generate_sft_data.py).

## Assumptions

- You already have a MemER release directory containing `train.json` and `media/`.
- You are using the official [`QwenLM/Qwen3-VL`](https://github.com/QwenLM/Qwen3-VL) finetuning code.

## 1. Clone Qwen3-VL

```bash
git clone https://github.com/QwenLM/Qwen3-VL.git
cd Qwen3-VL/qwen-vl-finetune
python -m pip install -e ../qwen-vl-utils
```

## 2. Register The Exported MemER Dataset

Qwen's finetune entrypoint expects local datasets to be registered in `qwenvl/data/__init__.py`.

Add an entry like this:

```python
MEMER_SFT = {
    "annotation_path": "/abs/path/to/release/dusting_train/train.json",
    "data_path": "/abs/path/to/release/dusting_train",
}
```

Then use the lowercase dataset name in `--dataset_use` in the training command. For example:

```text
--dataset_use memer_sft
```

## 3. Example Command

The command below is what we used to train on the 50 demo dusting dataset on 2 B200s: 

```bash
export QWEN_VL_ATTN_IMPL=flash_attention_2

torchrun \
  --nproc_per_node=2 \
  qwenvl/train/train_qwen.py \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --dataset_use memer_sft \
  --data_flatten True \
  --tune_mm_vision False \
  --tune_mm_mlp False \
  --tune_mm_llm True \
  --bf16 \
  --output_dir /abs/path/to/output \
  --num_train_epochs 15 \
  --max_steps 1500 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --optim adamw_torch \
  --max_pixels 115200 \
  --min_pixels 50176 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  --learning_rate 6e-5 \
  --weight_decay 0 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --logging_nan_inf_filter False \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8
```
