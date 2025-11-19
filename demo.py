#!/usr/bin/python3

import logging
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # noqa: E402
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments, set_seed)

from datasets import load_dataset

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# 差分隐私噪声相关函数
# ------------------------------------------------------

def matrix_gaussian_noise(epsilon: float, delta: float, sensitivity: float) -> float:
    """
    计算 Analytic Gaussian mechanism 的噪声强度

    Args:
        epsilon: 隐私预算
        delta: 失败概率 （通常是 1e-5）
        sensitivity: 查询的敏感度

    Returns:
        高斯分布的标准差
    """
    def function_phi(t):
        return (1 + math.erf(t / math.sqrt(2))) / 2

    def B_plus_function(v, eps):
        return function_phi(math.sqrt(eps * v)) - math.exp(eps) * \
               function_phi(-math.sqrt(eps * (v + 2)))

    def B_minus_function(u, eps):
        return function_phi(-math.sqrt(eps * u)) - math.exp(eps) * \
               function_phi(-math.sqrt(eps * (u + 2)))

    def compute_R(eps, delta_value, iterations=5000):
        delta_0 = function_phi(0) - math.exp(eps) * function_phi(-math.sqrt(2 * eps))
        start, end = 0, 1e5
        B_function = B_plus_function if delta_value >= delta_0 else B_minus_function
        for _ in range(iterations):
            mid = (start + end) / 2
            value = B_function(mid, eps)
            if value < delta_value:
                end = mid
            else:
                start = mid
        u_star = end
        if delta_value >= delta_0:
            alpha = math.sqrt(1 + u_star / 2) - math.sqrt(u_star / 2)
        else:
            alpha = math.sqrt(1 + u_star / 2) + math.sqrt(u_star / 2)
        return math.sqrt(2 * eps) / alpha

    R = compute_R(epsilon, delta)
    return sensitivity / R


def _max_norm_clip(embeddings: torch.Tensor, norm_c: float = 1.0) -> torch.Tensor:
    """
    对 embeddings 进行范数裁剪 （基于 token-level 定义）

    Args:
        embeddings: [batch_size, seq_len, hidden_dim]
        norm_c: 裁剪阈值

    Returns:
        裁剪后的 embeddings
    """
    # 计算每个样本的 L2 范数
    total_norms = torch.norm(embeddings, dim=-1, keepdim=True)
    # 计算裁剪系数（只裁剪超过 norm_c 的）
    clip_coef = norm_c / (total_norms + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    return embeddings * clip_coef


def add_noise_with_norm_control(
    embeddings: torch.Tensor,
    noise_factor: float,
    norm_c: float,
    add_noise: bool = True
) -> torch.Tensor:
    """
    添加 i.i.d. 高斯噪声

    Args:
        embeddings: 裁剪后的 embeddings
        noise_factor: 噪声强度（标准差）
        norm_c: 范数
        add_noise: 是否加噪

    Returns:
        加噪后的 embeddings
    """
    if not add_noise:
        return embeddings

    embeddings = _max_norm_clip(embeddings, norm_c)
    noise = torch.normal(
        mean=0.0,
        std=noise_factor,
        size=embeddings.shape,
        device=embeddings.device,
        dtype=embeddings.dtype
    )
    return embeddings + noise


# ------------------------------------------------------
# 用户：基本上只是定义一下用户的输入，没有什么处理过程
# ------------------------------------------------------
@dataclass
class UserClientConfig:
    train_file: str  # 训练数据集的路径
    validation_file: Optional[str] = None  # 验证数据集的路径
    model_name: str = "bert-base-uncased"  # 预训练模型的名称
    text_column: str = "sentence"  # 数据集文本对应标识
    label_column: str = "label"  # 数据集标签对应标识
    max_train_samples: Optional[int] = 200  # 训练样本的选择上限
    max_eval_samples: Optional[int] = 100  # 验证样本的选择上限


# 差分隐私相关参数格式
@dataclass
class DPConfig:
    epsilon: float = 8.0  # 隐私预算
    delta: float = 1e-5  # 错误概率
    norm_c: float = 5.0  # 裁剪阈值
    add_noise: bool = True  # 是否加噪
    add_noise_inference: bool = False  # 推理是否加噪


# ------------------------------------------------------
# 算安保：处理数据，将数据转换为 embeddings 形式，
#        加噪实际上是在 embeddings 上处理
# ------------------------------------------------------
@dataclass
class GatewayConfig:
    max_length: int = 128  # 填充长度（推荐值）
    dp_config: Optional[DPConfig] = None


@dataclass
class Gate2Compute:
    # 定义算安保的输出格式，用来传输给算力网
    model_name: str  # 预训练模型的名称
    payload: Dict  # 处理后的用户数据集
    # 以下是训练相关的参数
    output_dir: str = "output"  # 输出目录
    learning_rate: float = 2e-5  # 学习率
    weight_decay: float = 0.0
    num_train_epochs: float = 5.0  # 迭代次数
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    logging_steps: int = 10
    seed: int = 42


class PrivacyGateway:
    def __init__(self, user_config: UserClientConfig, gate_config: GatewayConfig):
        self.user_config = user_config
        self.gate_config = gate_config
        # 设置分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(user_config.model_name,
                                                       use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
                     user_config.model_name)
        self.model.eval()
        self.payload: Optional[Dict] = None

        self.dp_config = gate_config.dp_config
        # 如果需要添加噪声，初始化相关参数
        if self.dp_config and self.dp_config.add_noise:
            self._setup_noise_parameters()

    def _setup_noise_parameters(self):
        """设置噪声相关参数"""
        dp_cfg = self.dp_config
        if dp_cfg is None or not dp_cfg.add_noise:
            logger.info("[算安保]-噪声关闭，使用明文 embeddings（不加噪声）")
            return

        sensitivity = dp_cfg.norm_c
        self.noise_factor = matrix_gaussian_noise(
                            dp_cfg.epsilon,
                            dp_cfg.delta,
                            sensitivity)
        logger.info(f"[算安保]-DP配置：ε={dp_cfg.epsilon}, δ={dp_cfg.delta},\
norm_c={dp_cfg.norm_c}")
        logger.info(f"[算安保]-噪声因子 = {self.noise_factor:.4f}")

        signal_norm = dp_cfg.norm_c
        snr = signal_norm / (self.noise_factor + 1e-12 )
        logger.info(f"[算安保]-估计 SNR = {snr:.4f}")

    def load_data(self) -> Dict[str, Dict[str, List]]:
        # 分隔符，tsv 格式为 "\t"，csv 格式为 ","
        delimiter = "\t" if self.user_config.train_file.endswith(".tsv") else ","
        data_files = {"train": self.user_config.train_file}
        if self.user_config.validation_file:
            data_files["validation"] = self.user_config.validation_file
        logger.info(f"[算安保]-加载数据集：{data_files}")

        # 直接使用 load_dataset 对原始数据进行转换
        dataset = load_dataset("csv", data_files=data_files, delimiter=delimiter)

        train_data = self._subset(dataset, "train", self.user_config.max_train_samples)
        payload = {
            "text_column": self.user_config.text_column,
            "label_column": self.user_config.label_column,
            "splits": {"train": train_data}
        }
        if "validation" in dataset:
            eval_data = self._subset(dataset, "validation",
                                     self.user_config.max_eval_samples)
            payload["splits"]["validation"] = eval_data
        # 根据用户设置的样本大小，选定的最终数据集
        return payload

    def sanitize(self, raw_payload: Dict) -> Dict:
        # 对 raw payload 转换为 embeddings，然后加噪
        text_col = raw_payload["text_column"]
        label_col = raw_payload["label_column"]

        sanitized_payload = {"splits": {}}

        for split_name, split_data in raw_payload["splits"].items():
            texts = split_data[text_col]
            labels = split_data[label_col]

            # sanitized_split = self._encode_split(texts)
            # sanitized_split["labels"] = labels
            sanitized_payload["splits"][split_name] = self._encode_split(texts, labels)

            logger.info(
                f"[算安保]-处理 {split_name} 数据集 -> embeddings 形式"
                f"{' (已加噪)' if self.dp_config and self.dp_config.add_noise else ''}"
            )

        self.payload = sanitized_payload
        return sanitized_payload

    def output(self) -> Gate2Compute:
        if self.payload is None:
            raise ValueError("请先调用 sanitize() 生成加噪后的 payload")
        return Gate2Compute(model_name=self.user_config.model_name,
                            payload=self.payload)

    def _subset(self, dataset, split: str, limit: Optional[int]) -> Dict[str, List]:
        data = dataset[split]
        if limit is not None:
            limit = min(limit, len(data))
            data = data.select(range(limit))
        return data.to_dict()

    # @torch.no_grad()
    def _encode_split(self, texts: List[str], labels: List[int]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            encodings = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.gate_config.max_length,
                return_tensors="pt"
            )
            base_model = self.model.base_model
            outputs = base_model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                token_type_ids=encodings.get("token_type_ids"),
                output_hidden_states=True,
                return_dict=True
            )
            embeddings = outputs.hidden_states[0]

            # 添加噪声
            if self.dp_config and self.dp_config.add_noise:
                embeddings = add_noise_with_norm_control(
                    embeddings=embeddings,
                    noise_factor=self.noise_factor,
                    norm_c=self.dp_config.norm_c,
                    add_noise=True
                )
        return {
            "embeddings": embeddings.cpu(),
            "attention_mask": encodings["attention_mask"].cpu(),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


# ------------------------------------------------------
# 算力网：接收 embeddings，继续执行微调
# ------------------------------------------------------

class EmbeddingDataset(TorchDataset):
    def __init__(self, split: Dict[str, torch.Tensor]):
        self.embeddings = split["embeddings"].float()
        self.attention_mask = split["attention_mask"].long()
        self.labels = split["labels"].long()

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return {
            "inputs_embeds": self.embeddings[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }


class ComputeServer:
    def __init__(self, config: Gate2Compute):
        self.config = config

    def _build_dataset(self, split: Dict[str, torch.Tensor]) -> EmbeddingDataset:
        return EmbeddingDataset(split)

    def fine_tune(self) -> Dict[str, float]:
        set_seed(self.config.seed)

        splits_dataset = self.config.payload["splits"]
        train_dataset = self._build_dataset(splits_dataset["train"])

        eval_flag = "validation" in splits_dataset
        eval_dataset = None
        if eval_flag:
            eval_dataset = self._build_dataset(splits_dataset["validation"])

        config = AutoConfig.from_pretrained(
            self.config.model_name,
            num_labels=len(torch.unique(train_dataset.labels))
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            config=config
        )

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            num_train_epochs=self.config.num_train_epochs,
            logging_steps=self.config.logging_steps,
            eval_strategy="epoch" if eval_flag else "no",
            save_strategy="no",
            report_to="none",
            seed=self.config.seed
        )

        def compute_metrics(pred):
            logits = pred.predictions
            labels = pred.label_ids
            preds = np.argmax(logits, axis=1)
            accuracy = (preds == labels).mean()
            return {"accuracy": float(accuracy)}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_flag else None,
            compute_metrics=compute_metrics if eval_flag else None,
        )

        logger.info(f"[算力网]-开始微调，训练样本规模： {len(train_dataset)}")
        train_result = trainer.train()

        metrics = {"train_loss": train_result.training_loss}

        if eval_flag:
            logger.info(f"[算力网]-开始验证")
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)
            logger.info(f"验证结果：{eval_metrics}")

        return metrics


def run_pipeline(user_config: UserClientConfig,
                 gate_config: GatewayConfig, title: str) -> Dict[str, float]:
    logger.info("\n" + "=" * 60)
    logger.info(title)
    logger.info("=" * 60)

    gateway = PrivacyGateway(user_config=user_config, gate_config=gate_config)
    raw_payload = gateway.load_data()
    gateway.sanitize(raw_payload)
    output = gateway.output()

    server = ComputeServer(output)
    metrics = server.fine_tune()
    logger.info(f"{title} 结果：{metrics}")
    return metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s : %(message)s", force=True)

    # ========== 配置参数 ==========
    user_config = UserClientConfig(
        train_file="datasets/SST-2/train.tsv",
        validation_file="datasets/SST-2/dev.tsv",
        model_name="bert-base-uncased",
        max_train_samples=20,   # 快速测试用少量样本
        max_eval_samples=10,
    )

    baseline_metrics = run_pipeline(
        user_config,
        gate_config=GatewayConfig(max_length=128, dp_config=None),
        title="测试 1：不加噪测试"
    )

    dp_metrics = run_pipeline(
        user_config,
        gate_config=GatewayConfig(
            max_length=128,
            dp_config=DPConfig(epsilon=8.0, delta=1e-5, norm_c=1.0, add_noise=True)
        ),
        title="测试 2：添加 DP 噪声"
    )
