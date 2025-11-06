#!/usr/bin/python3

import logging
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
def _compute_noise_multiplier(epsilon, delta, sensitivity):
    """
    计算高斯噪声的 noise_multiplier (基于 analytic Gaussian mechanism)

    Args:
        epsilon: 隐私预算
        delta: 隐私参数
        sensitivity: 敏感度 (裁剪后为 2 * norm_c)

    Returns:
        noise_b: 噪声标准差
    """
    # 使用 analytic Gaussian mechanism 的简化公式
    # 对于 local DP，我们使用标准高斯机制的公式
    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise_b = sensitivity * sigma
    return noise_b


def _max_norm_clip(embeddings, norm_c=1.0):
    """
    对 embeddings 进行范数裁剪

    Args:
        embeddings: [batch_size, seq_len, hidden_dim]
        norm_c: 裁剪阈值

    Returns:
        裁剪后的 embeddings
    """
    shape = embeddings.shape
    # 将每个样本展平为一维向量
    embeddings_flat = embeddings.reshape(shape[0], -1)

    # 计算每个样本的 L2 范数
    total_norm = torch.norm(embeddings_flat, dim=-1, keepdim=True)

    # 计算裁剪系数（只裁剪超过 norm_c 的）
    clip_coef = norm_c / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    # 应用裁剪
    embeddings_flat = embeddings_flat * clip_coef_clamped

    return embeddings_flat.reshape(shape)


def _add_gaussian_noise(embeddings, noise_multiplier):
    """
    添加高斯噪声

    Args:
        embeddings: 裁剪后的 embeddings
        noise_multiplier: 噪声强度（标准差）

    Returns:
        加噪后的 embeddings
    """
    noise = torch.randn_like(embeddings) * noise_multiplier
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
    max_train_samples: Optional[int] = None  # 训练样本的选择上限
    max_eval_samples: Optional[int] = None  # 验证样本的选择上限


# ------------------------------------------------------
# 算安保：处理数据，将数据转换为 embeddings 形式，
#        加噪实际上是在 embeddings 上处理
# ------------------------------------------------------
@dataclass
class GatewayConfig:
    max_length: int = 128  # 填充长度（推荐值）
    add_noise: bool = False  # 是否添加噪声
    auto_norm_c: bool = False  # 是否自动估计 norm_c（默认使用论文推荐值）
    epsilon: float = 8.0  # 隐私预算（默认值）
    delta: float = 1e-5  # 隐私参数
    norm_c: float = 1.0  # 裁剪阈值（论文推荐值，如果 auto_norm_c=True 会被覆盖）
    norm_percentile: int = 50  # 自动估计时使用第几百分位（推荐使用中位数）


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

        # 如果需要添加噪声，初始化相关参数
        if gate_config.add_noise:
            self._setup_noise_parameters()

    def _setup_noise_parameters(self):
        """设置噪声相关参数"""
        logger.info(f"[算安保]-配置差分隐私参数:")
        logger.info(f"  epsilon = {self.gate_config.epsilon}")
        logger.info(f"  delta = {self.gate_config.delta}")

        # 如果需要自动估计 norm_c，先用少量样本估计
        if self.gate_config.auto_norm_c:
            logger.info(f"  启用自动估计 norm_c...")
            # 注意：这里会在第一次调用 sanitize 时进行估计
            self._norm_c_estimated = False
        else:
            logger.info(f"  norm_c = {self.gate_config.norm_c}")
            self._norm_c_estimated = True
            self._compute_noise_params()

    def _estimate_norm_c(self, sample_texts: List[str]):
        """
        基于样本数据估计合适的 norm_c

        Args:
            sample_texts: 用于估计的文本样本
        """
        if self._norm_c_estimated:
            return

        logger.info("[算安保]-正在估计合适的 norm_c（使用100个样本）...")

        # 限制样本数量以加快估计
        sample_texts = sample_texts[:100]

        # 获取 embeddings（不加噪声）
        with torch.no_grad():
            encodings = self.tokenizer(
                sample_texts,
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

            # 计算每个样本的范数
            embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
            norms = torch.norm(embeddings_flat, dim=-1).cpu().numpy()

        # 使用指定百分位作为 norm_c
        suggested_norm_c = float(np.percentile(norms, self.gate_config.norm_percentile))

        # 统计信息
        median_norm = float(np.median(norms))
        max_norm = float(np.max(norms))
        clipped_ratio = (norms > suggested_norm_c).mean() * 100

        logger.info(f"[算安保]-范数统计：")
        logger.info(f"  中位数: {median_norm:.4f}")
        logger.info(f"  {self.gate_config.norm_percentile}th percentile: {suggested_norm_c:.4f}")
        logger.info(f"  最大值: {max_norm:.4f}")
        logger.info(f"  将被裁剪的样本比例: {clipped_ratio:.1f}%")
        logger.info(f"  自动设置 norm_c = {suggested_norm_c:.4f}")

        # 警告：如果 norm_c 太大，会导致噪声过大
        if suggested_norm_c > 50:
            logger.warning(
                f"  ⚠️  自动估计的 norm_c={suggested_norm_c:.1f} 过大！\n"
                f"  这会导致噪声强度过大（noise ∝ norm_c），SNR 可能很低。\n"
                f"  建议：\n"
                f"    1. 使用论文推荐的 norm_c=1.0（更激进的裁剪，但噪声小得多）\n"
                f"    2. 或使用 norm_percentile=50（中位数）而非 {self.gate_config.norm_percentile}"
            )

        # 更新配置
        self.gate_config.norm_c = suggested_norm_c
        self._norm_c_estimated = True

        # 计算噪声参数
        self._compute_noise_params()

    def _compute_noise_params(self):
        """计算噪声强度"""
        # 敏感度 = 2 * norm_c（两个样本最多差异）
        sensitivity = 2.0 * self.gate_config.norm_c

        # 计算噪声强度
        self.noise_multiplier = _compute_noise_multiplier(
            epsilon=self.gate_config.epsilon,
            delta=self.gate_config.delta,
            sensitivity=sensitivity
        )

        logger.info(f"[算安保]-噪声参数:")
        logger.info(f"  sensitivity = {sensitivity:.4f}")
        logger.info(f"  noise_multiplier = {self.noise_multiplier:.4f}")

        # 估计 SNR（信噪比）
        n = self.gate_config.max_length
        d = 768  # BERT-base hidden size
        signal_per_entry = self.gate_config.norm_c / np.sqrt(n * d)
        noise_per_entry = self.noise_multiplier / np.sqrt(n * d)
        snr = signal_per_entry / noise_per_entry

        logger.info(f"  估计 SNR = {snr:.4f}")
        if snr < 0.5:
            logger.warning(f"  ⚠️  SNR 太低！考虑增大 epsilon 或减小 norm_c")
        elif snr < 1.0:
            logger.warning(f"  ⚠️  SNR 较低，准确度可能明显下降")
        elif snr < 2.0:
            logger.info(f"  ✓  SNR 适中")
        else:
            logger.info(f"  ✓  SNR 良好")

    def load_data(self):
        # 分隔符，tsv 格式为 "\t"，csv 格式为 ","
        delimiter = "\t" if self.user_config.train_file.endswith(".tsv") else ","
        data_files = {"train": self.user_config.train_file}
        if self.user_config.validation_file is not None:
            data_files.setdefault("validation", self.user_config.validation_file)
        logger.info(f"[算安保]-加载数据集：{data_files}")
        # 直接使用 load_dataset 对原始数据进行转换
        dataset = load_dataset("csv", data_files=data_files, delimiter=delimiter)

        train_data = self._subset(dataset, "train", self.user_config.max_train_samples)
        payload = {
            "text_column": self.user_config.text_column,
            "label_column": self.user_config.label_column,
            "splits": {
                "train": train_data
            }
        }
        if "validation" in dataset:
            eval_data = self._subset(dataset, "validation",
                                     self.user_config.max_eval_samples)
            payload["splits"].setdefault("validation", eval_data)
        # 根据用户设置的样本大小，选定的最终数据集
        return payload

    def sanitize(self, payload: Dict) -> Dict:
        # 对 payload 转换为 embeddings，然后加噪
        text_col = payload["text_column"]
        label_col = payload["label_column"]

        sanitized_payload = {"splits": {}}

        for split_name, split_data in payload["splits"].items():
            texts = split_data[text_col]
            labels = torch.tensor(split_data[label_col], dtype=torch.long)

            # 如果启用自动估计且还未估计，用训练集估计 norm_c
            if (self.gate_config.add_noise and
                self.gate_config.auto_norm_c and
                not self._norm_c_estimated and
                split_name == "train"):
                self._estimate_norm_c(texts)

            sanitized_split = self._encode_split(texts)
            sanitized_split["labels"] = labels
            sanitized_payload["splits"][split_name] = sanitized_split

            logger.info(
                f"[算安保]-处理 {split_name} 数据集 -> embeddings 形式"
                f"{' (已加噪)' if self.gate_config.add_noise else ''}"
            )

        self.payload = sanitized_payload
        return sanitized_payload

    def output(self) -> Gate2Compute:
        output_content = Gate2Compute(model_name=self.user_config.model_name,
                                      payload=self.payload)
        return output_content

    def _subset(self, dataset: Dict, split: str, limit: Optional[int]) -> Dict[str, List]:
        data = dataset[split]
        if limit is not None:
            limit = min(limit, len(data))
            data = data.select(range(limit))
        return data.to_dict()

    @torch.no_grad()
    def _encode_split(self, texts: List[str]) -> Dict[str, torch.Tensor]:
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
        if self.gate_config.add_noise:
            # Step 1: 范数裁剪
            embeddings = _max_norm_clip(embeddings, self.gate_config.norm_c)

            # Step 2: 添加高斯噪声
            embeddings = _add_gaussian_noise(embeddings, self.noise_multiplier)

        sanitized = {
            "embeddings": embeddings.cpu(),  # [batch, seq, hidden]
            "attention_mask": encodings["attention_mask"].cpu()
        }
        return sanitized


# ------------------------------------------------------
# 算力网：接收 embeddings，继续执行微调
# ------------------------------------------------------

class EmbeddingDataset(TorchDataset):
    def __init__(self, embeddings: torch.Tensor, attention_mask: torch.Tensor,
                 labels: torch.Tensor):
        self.embeddings = embeddings.float()
        self.attention_mask = attention_mask.long()
        self.labels = labels.long()

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
        return EmbeddingDataset(
            embeddings=split["embeddings"],
            attention_mask=split["attention_mask"],
            labels=split["labels"]
        )

    def fine_tune(self) -> Dict:
        set_seed(self.config.seed)

        train_dataset = self._build_dataset(self.config.payload["splits"]["train"])

        eval_flag = False
        eval_dataset = None
        if "validation" in self.config.payload["splits"]:
            eval_flag = True
            eval_dataset = self._build_dataset(self.config.payload["splits"]["validation"])

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
            save_strategy="epoch",
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
            tokenizer=None,  # 我们直接传入 embeddings，不需要 tokenizer
            data_collator=None,  # 数据集已是张量，默认 collate 足够
            compute_metrics=compute_metrics if eval_flag else None,
        )

        logger.info(f"[算力网]-开始进行微调，样本规模为 {len(train_dataset)}")
        train_result = trainer.train()

        metrics = {"train_loss": train_result.training_loss}

        if eval_flag:
            logger.info(f"[算力网]-开始验证")
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)
            logger.info(f"验证结果：{eval_metrics}")

        return metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s : %(message)s")

    # ========== 配置参数 ==========
    user_config = UserClientConfig(
        train_file="datasets/SST-2/train.tsv",
        validation_file="datasets/SST-2/dev.tsv",
        model_name="bert-base-uncased",
        max_train_samples=200,   # 快速测试用少量样本
        max_eval_samples=100,
    )

    # 测试1: 不加噪声（基线）
    logger.info("\n" + "="*60)
    logger.info("测试 1: 不添加噪声（基线）")
    logger.info("="*60)
    gateway_no_noise = PrivacyGateway(
        user_config=user_config,
        gate_config=GatewayConfig(add_noise=False)
    )
    raw_payload = gateway_no_noise.load_data()
    sanitized_payload = gateway_no_noise.sanitize(raw_payload)
    output_content = gateway_no_noise.output()

    server = ComputeServer(output_content)
    metrics_no_noise = server.fine_tune()

    # 测试2: 添加噪声（论文推荐：norm_c=1.0）
    logger.info("\n" + "="*60)
    logger.info("测试 2: 添加 DP 噪声 - 论文推荐方法 (norm_c=1.0, ε=8.0)")
    logger.info("="*60)
    gateway_paper_default = PrivacyGateway(
        user_config=user_config,
        gate_config=GatewayConfig(
            add_noise=True,
            auto_norm_c=False,  # 使用论文推荐的固定值
            norm_c=1.0,         # 论文推荐值
            epsilon=8.0,
            delta=1e-5,
        )
    )
    raw_payload = gateway_paper_default.load_data()
    sanitized_payload = gateway_paper_default.sanitize(raw_payload)
    output_content = gateway_paper_default.output()

    server = ComputeServer(output_content)
    metrics_paper = server.fine_tune()

    # 测试3: 添加噪声（自适应：auto norm_c with median）
    logger.info("\n" + "="*60)
    logger.info("测试 3: 添加 DP 噪声 - 自适应方法 (auto norm_c, 中位数, ε=8.0)")
    logger.info("="*60)
    gateway_adaptive = PrivacyGateway(
        user_config=user_config,
        gate_config=GatewayConfig(
            add_noise=True,
            auto_norm_c=True,       # 自动估计
            norm_percentile=50,     # 使用中位数（而非90分位）
            epsilon=8.0,
            delta=1e-5,
        )
    )
    raw_payload = gateway_adaptive.load_data()
    sanitized_payload = gateway_adaptive.sanitize(raw_payload)
    output_content = gateway_adaptive.output()

    server = ComputeServer(output_content)
    metrics_adaptive = server.fine_tune()

    # 对比结果
    logger.info("\n" + "="*70)
    logger.info("最终结果对比")
    logger.info("="*70)
    logger.info(f"【基线】无噪声:              {metrics_no_noise}")
    logger.info(f"【方法1】论文推荐(norm_c=1): {metrics_paper}")
    logger.info(f"【方法2】自适应(中位数):     {metrics_adaptive}")

    if "eval_accuracy" in metrics_no_noise:
        baseline_acc = metrics_no_noise["eval_accuracy"]
        logger.info("\n准确度对比：")
        logger.info(f"  基线准确度: {baseline_acc*100:.2f}%")

        if "eval_accuracy" in metrics_paper:
            paper_acc = metrics_paper["eval_accuracy"]
            paper_drop = (baseline_acc - paper_acc) * 100
            logger.info(f"  论文方法:   {paper_acc*100:.2f}% (下降 {paper_drop:.2f}%)")

        if "eval_accuracy" in metrics_adaptive:
            adaptive_acc = metrics_adaptive["eval_accuracy"]
            adaptive_drop = (baseline_acc - adaptive_acc) * 100
            logger.info(f"  自适应方法: {adaptive_acc*100:.2f}% (下降 {adaptive_drop:.2f}%)")

        logger.info("\n💡 建议: 如果论文方法准确度可接受，优先使用 norm_c=1.0（噪声更小，隐私保证更强）")
