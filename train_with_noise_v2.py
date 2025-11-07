#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DP-Forward 集中式训练脚本 v2
使用预计算的带噪 embeddings（每个样本的噪声固定）
"""

import math

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments, set_seed)

from datasets import load_dataset

# ============================================================
# 噪声函数
# ============================================================

def _compute_noise_multiplier(epsilon, delta, sensitivity):
    """计算高斯噪声的标准差"""
    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise_multiplier = sensitivity * sigma
    return noise_multiplier


def _max_norm_clip(embeddings, norm_c=1.0):
    """范数裁剪：将 embeddings 的范数限制在 norm_c"""
    shape = embeddings.shape
    embeddings_flat = embeddings.reshape(shape[0], -1)
    total_norm = torch.norm(embeddings_flat, dim=-1, keepdim=True)
    clip_coef = norm_c / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    embeddings_flat = embeddings_flat * clip_coef_clamped
    return embeddings_flat.reshape(shape)


def _add_gaussian_noise(embeddings, noise_multiplier):
    """添加高斯噪声（只调用一次！）"""
    noise = torch.randn_like(embeddings) * noise_multiplier
    return embeddings + noise


# ============================================================
# 预计算带噪 Embeddings
# ============================================================

@torch.no_grad()
def precompute_embeddings(model, dataset, tokenizer, noise_position,
                         epsilon, delta, norm_c, device, batch_size=32):
    """
    预计算所有样本的 embeddings 并添加固定的噪声

    Args:
        model: BERT 模型
        dataset: 原始数据集
        tokenizer: 分词器
        noise_position: 加噪位置 ("none", "embedding", "encoder_1")
        epsilon, delta, norm_c: 隐私参数
        device: 设备
        batch_size: 批大小

    Returns:
        noisy_embeddings: 带噪的 embeddings 列表
        attention_masks: attention_mask 列表
        labels: 标签列表
    """
    model.eval()
    model.to(device)

    # 计算噪声参数
    if noise_position != "none":
        sensitivity = 2 * norm_c
        noise_multiplier = _compute_noise_multiplier(epsilon, delta, sensitivity)

        # 计算 SNR
        n = 128
        d = 768
        signal_per_entry = norm_c / np.sqrt(n * d)
        noise_per_entry = noise_multiplier / np.sqrt(n * d)
        snr = signal_per_entry / noise_per_entry

        print(f"\n[Noise Parameters]")
        print(f"  ε (epsilon): {epsilon}")
        print(f"  δ (delta): {delta}")
        print(f"  norm_c: {norm_c}")
        print(f"  noise_multiplier: {noise_multiplier:.4f}")
        print(f"  SNR: {snr:.4f}")
        if snr < 1.0:
            print(f"  ⚠️  SNR < 1，噪声比信号大 {1/snr:.2f} 倍")
    else:
        noise_multiplier = 0
        print(f"\n[No Noise - Baseline]")

    all_embeddings = []
    all_attention_masks = []
    all_labels = []

    # 获取 base model
    base_model = model.base_model

    print(f"\nPrecomputing embeddings for {len(dataset)} samples...")

    # 分批处理
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:min(i+batch_size, len(dataset))]

        # 分词
        texts = batch["sentence"]
        inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # 前向传播获取 embeddings
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 根据位置选择 embeddings
        if noise_position == "embedding":
            embeddings = outputs.hidden_states[0]  # [batch, seq_len, hidden_dim]
        elif noise_position == "encoder_1":
            embeddings = outputs.hidden_states[1]
        else:  # "none"
            embeddings = outputs.hidden_states[0]

        # 添加噪声（只添加一次！）
        if noise_position != "none":
            embeddings = _max_norm_clip(embeddings, norm_c)
            embeddings = _add_gaussian_noise(embeddings, noise_multiplier)

        # 保存
        all_embeddings.append(embeddings.cpu())
        all_attention_masks.append(attention_mask.cpu())
        all_labels.extend(batch["label"])

    # 合并
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)
    all_labels = torch.tensor(all_labels)

    print(f"Precomputation done! Shape: {all_embeddings.shape}")

    return all_embeddings, all_attention_masks, all_labels


# ============================================================
# 自定义 Dataset（使用预计算的 embeddings）
# ============================================================

class EmbeddingDataset(Dataset):
    """使用预计算的 embeddings 的数据集"""

    def __init__(self, embeddings, attention_masks, labels):
        self.embeddings = embeddings
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "inputs_embeds": self.embeddings[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx]
        }


# ============================================================
# 训练和评估
# ============================================================

def train_and_evaluate(model, train_dataset, eval_dataset, output_dir,
                       epochs=3, batch_size=32, learning_rate=2e-5):
    """训练并评估模型"""

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        disable_tqdm=False,
        report_to="none",
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda eval_pred: {
            "accuracy": (eval_pred.predictions.argmax(-1) == eval_pred.label_ids).mean()
        }
    )

    # 训练
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}")
    trainer.train()

    # 评估
    eval_results = trainer.evaluate()

    return eval_results["eval_accuracy"]


# ============================================================
# 主函数
# ============================================================

def main():
    # 设置随机种子
    set_seed(42)

    # 配置
    model_name = "bert-base-uncased"
    epochs = 3
    batch_size = 32
    learning_rate = 2e-5
    delta = 1e-5
    norm_c = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 样本数量限制
    max_train_samples = None  # 使用全部数据
    max_eval_samples = None

    # 测试配置
    noise_position = "embedding"  # 只测试 embedding 层
    epsilon = 128

    print(f"Device: {device}")
    print(f"{'='*80}")

    # 加载数据集
    print("Loading dataset...")
    dataset = load_dataset("glue", "sst2")

    train_raw = dataset["train"]
    eval_raw = dataset["validation"]

    if max_train_samples is not None:
        train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))
    if max_eval_samples is not None:
        eval_raw = eval_raw.select(range(min(max_eval_samples, len(eval_raw))))

    print(f"Train samples: {len(train_raw)}")
    print(f"Eval samples: {len(eval_raw)}")

    # 加载 tokenizer 和模型
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # 预计算训练集的 embeddings
    print(f"\n{'='*80}")
    print(f"Precomputing TRAIN embeddings (noise_position={noise_position}, ε={epsilon})")
    print(f"{'='*80}")
    train_embeddings, train_masks, train_labels = precompute_embeddings(
        model, train_raw, tokenizer, noise_position,
        epsilon, delta, norm_c, device, batch_size=batch_size
    )

    # 预计算验证集的 embeddings（使用相同的噪声参数）
    print(f"\n{'='*80}")
    print(f"Precomputing EVAL embeddings (noise_position={noise_position}, ε={epsilon})")
    print(f"{'='*80}")
    eval_embeddings, eval_masks, eval_labels = precompute_embeddings(
        model, eval_raw, tokenizer, noise_position,
        epsilon, delta, norm_c, device, batch_size=batch_size
    )

    # 创建 Dataset
    train_dataset = EmbeddingDataset(train_embeddings, train_masks, train_labels)
    eval_dataset = EmbeddingDataset(eval_embeddings, eval_masks, eval_labels)

    # 重新加载模型用于训练（清除之前的状态）
    print("\nReloading model for training...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Freeze embedding 层（因为已经预计算了）
    print("Freezing embedding layers...")
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False

    # 如果是 encoder_1，还要 freeze encoder 0
    if noise_position == "encoder_1":
        for param in model.base_model.encoder.layer[0].parameters():
            param.requires_grad = False

    # 训练
    accuracy = train_and_evaluate(
        model, train_dataset, eval_dataset,
        output_dir=f"./output/sst2_{noise_position}_eps{epsilon}",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # 打印结果
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Noise Position: {noise_position}")
    print(f"Epsilon: {epsilon}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
