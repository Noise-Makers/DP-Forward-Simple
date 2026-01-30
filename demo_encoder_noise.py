#!/usr/bin/python3
"""
一个对照 demo：在 encoder 层加噪（DP-Forward 思路），保持“用户-算安保-算力”角色划分。
与 demo.py 不同点：
- 不预先把 embeddings 加噪；算力端在 BERT encoder 指定层的输出处加噪。
- 噪声通过 forward hook 注入，逻辑与 simple_dp_forward_encoder.py 类似。

用途：快速理解编码器加噪的流程，而不改动仓库其他代码。
"""

import inspect
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments, set_seed)

from dp_noise import add_noise, get_noise_multiplier

logger = logging.getLogger(__name__)


# ========================= 基础配置 =========================
@dataclass
class UserClientConfig:
    train_file: str
    validation_file: Optional[str] = None
    model_name: str = "bert-base-uncased"
    text_column: str = "sentence"
    label_column: str = "label"
    max_train_samples: Optional[int] = 200
    max_eval_samples: Optional[int] = 100


@dataclass
class DPConfig:
    epsilon: float = 8.0
    delta: float = 1e-5
    norm_c: float = 1.0
    add_noise: bool = True
    add_noise_inference: bool = False  # 对于 embeddings，这里通常 False
    noise_factor: Optional[float] = None  # 若给定则跳过会计
    effective_batch_size: int = 32
    noise_layer: int = 10  # 保留字段但不再使用（改为安保端加噪）


@dataclass
class GatewayConfig:
    max_length: int = 128
    dp_config: Optional[DPConfig] = None


@dataclass
class Gate2Compute:
    model_name: str
    payload: Dict
    output_dir: str = "output"
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    logging_steps: int = 10
    seed: int = 42


# ========================= 算安保：只做 tokenization，不加噪 =========================
class PrivacyGateway:
    """算安保：分词 + 直接在 embedding 输出处加噪，传递 inputs_embeds 给算力端。"""

    def __init__(self, user_config: UserClientConfig, gate_config: GatewayConfig):
        self.user_config = user_config
        self.gate_config = gate_config
        self.tokenizer = AutoTokenizer.from_pretrained(user_config.model_name, use_fast=True)
        # 用于抽取 embeddings
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            user_config.model_name
        ).base_model
        self.base_model.eval()

    def _compute_noise_factor(self, dp_cfg: DPConfig, train_size: int) -> float:
        if not dp_cfg.add_noise:
            return 0.0
        if dp_cfg.noise_factor is not None:
            return dp_cfg.noise_factor
        return get_noise_multiplier(
            eps=dp_cfg.epsilon,
            delta=dp_cfg.delta,
            batch_size=dp_cfg.effective_batch_size,
            dataset_size=train_size,
            epoch=1,  # 粗略估计，用 1 个 epoch；如需更精确可传实际 epoch
            local_dp=True,
            noise_type="aGM",
        )

    def load_data(self) -> Dict[str, Dict[str, List]]:
        delimiter = "\t" if self.user_config.train_file.endswith(".tsv") else ","
        data_files = {"train": self.user_config.train_file}
        if self.user_config.validation_file:
            data_files["validation"] = self.user_config.validation_file
        dataset = load_dataset("csv", data_files=data_files, delimiter=delimiter)

        def subset(split, limit):
            data = dataset[split]
            if limit is not None:
                data = data.select(range(min(limit, len(data))))
            return data.to_dict()

        payload = {
            "text_column": self.user_config.text_column,
            "label_column": self.user_config.label_column,
            "splits": {"train": subset("train", self.user_config.max_train_samples)},
        }
        if "validation" in dataset:
            payload["splits"]["validation"] = subset("validation", self.user_config.max_eval_samples)
        return payload

    def tokenize_and_add_noise(self, raw_payload: Dict) -> Dict:
        text_col, label_col = raw_payload["text_column"], raw_payload["label_column"]
        tokenized = {"splits": {}}
        dp_cfg = self.gate_config.dp_config

        # 估算噪声
        train_size = len(raw_payload["splits"]["train"][text_col])
        noise_factor = self._compute_noise_factor(dp_cfg, train_size) if dp_cfg else 0.0
        if dp_cfg and dp_cfg.add_noise:
            logger.info(f"[算安保]-noise_factor={noise_factor:.4f} (eps={dp_cfg.epsilon}, delta={dp_cfg.delta})")

        for split_name, split_data in raw_payload["splits"].items():
            enc = self.tokenizer(
                split_data[text_col],
                padding="max_length",
                truncation=True,
                max_length=self.gate_config.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    token_type_ids=enc.get("token_type_ids"),
                    output_hidden_states=True,
                    return_dict=True,
                )
                embeddings = outputs.hidden_states[0]
                if dp_cfg and dp_cfg.add_noise:
                    embeddings = add_noise(
                        embeddings=embeddings,
                        noise_factor=noise_factor,
                        norm_c=dp_cfg.norm_c,
                        add_noise=True,
                    )

            tokenized["splits"][split_name] = {
                "inputs_embeds": embeddings.cpu(),
                "attention_mask": enc["attention_mask"].cpu(),
                "labels": torch.tensor(split_data[label_col], dtype=torch.long),
            }
        return tokenized

    def output(self, tokenized_payload: Dict) -> Gate2Compute:
        return Gate2Compute(model_name=self.user_config.model_name, payload=tokenized_payload)


# ========================= 算力：在 encoder 指定层加噪，继续训练 =========================
class TokenDataset(TorchDataset):
    def __init__(self, split: Dict[str, torch.Tensor]):
        self.split = split
    def __len__(self):
        return self.split["inputs_embeds"].size(0)
    def __getitem__(self, idx):
        return {
            "inputs_embeds": self.split["inputs_embeds"][idx],
            "attention_mask": self.split["attention_mask"][idx],
            "labels": self.split["labels"][idx],
        }


class ComputeServer:
    def __init__(self, config: Gate2Compute, dp_config: Optional[DPConfig]):
        self.config = config
        self.dp_config = dp_config  # 此时仅用于记录；不再在算力端加噪

    def _build_dataset(self, split: Dict[str, torch.Tensor]) -> TorchDataset:
        return TokenDataset(split)

    def fine_tune(self) -> Dict[str, float]:
        set_seed(self.config.seed)
        splits = self.config.payload["splits"]
        train_ds = self._build_dataset(splits["train"])
        eval_ds = self._build_dataset(splits["validation"]) if "validation" in splits else None

        config = AutoConfig.from_pretrained(self.config.model_name,
                                            num_labels=len(torch.unique(train_ds.split["labels"])))
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, config=config)

        # version-tolerant TrainingArguments
        ta_kwargs = dict(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            num_train_epochs=self.config.num_train_epochs,
            logging_steps=self.config.logging_steps,
            report_to="none",
            seed=self.config.seed,
        )
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        has_eval_strategy = "evaluation_strategy" in ta_params
        has_eval_during = "evaluate_during_training" in ta_params

        if has_eval_strategy:
            ta_kwargs["evaluation_strategy"] = "epoch" if eval_ds else "no"
            if "save_strategy" in ta_params:
                ta_kwargs["save_strategy"] = "no"
        elif has_eval_during:
            ta_kwargs["evaluate_during_training"] = bool(eval_ds)
            if "eval_steps" in ta_params:
                ta_kwargs["eval_steps"] = self.config.logging_steps
            if "save_strategy" in ta_params:
                ta_kwargs["save_strategy"] = "no"

        args = TrainingArguments(**ta_kwargs)

        def compute_metrics(pred):
            logits = pred.predictions
            labels = pred.label_ids
            preds = np.argmax(logits, axis=1)
            return {"accuracy": float((preds == labels).mean())}

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics if eval_ds else None,
        )

        trainer.train()
        metrics = {}
        if eval_ds:
            metrics.update(trainer.evaluate())
        return metrics


# ========================= 组合流水线 =========================
def run_pipeline(user_cfg: UserClientConfig, gate_cfg: GatewayConfig, title: str):
    logger.info("\n" + "=" * 50 + f"\n{title}\n" + "=" * 50)
    gateway = PrivacyGateway(user_cfg, gate_cfg)
    raw = gateway.load_data()
    tokenized = gateway.tokenize(raw)
    output = gateway.output(tokenized)

    server = ComputeServer(output, gate_cfg.dp_config)
    metrics = server.fine_tune()
    logger.info(f"{title} 结果: {metrics}")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)

    user_cfg = UserClientConfig(
        train_file="datasets/SST-2/train.tsv",
        validation_file="datasets/SST-2/dev.tsv",
        model_name="bert-base-uncased",
        max_train_samples=200,
        max_eval_samples=100,
    )

    # baseline
    run_pipeline(
        user_cfg,
        gate_cfg=GatewayConfig(max_length=128, dp_config=None),
        title="测试1：不加噪（baseline）",
    )

    # encoder 加噪
    dp_cfg = DPConfig(
        epsilon=8.0,
        delta=1e-5,
        norm_c=1.0,
        add_noise=True,
        noise_factor=None,  # 若想手设噪声，如 0.02，可直接填值
        effective_batch_size=32,
        noise_layer=10,
    )
    run_pipeline(
        user_cfg,
        gate_cfg=GatewayConfig(max_length=128, dp_config=dp_cfg),
        title="测试2：encoder 加噪（DP-Forward）",
    )
