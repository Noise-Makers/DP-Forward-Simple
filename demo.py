#!/usr/bin/python3

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments, set_seed)

from datasets import load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 用户：负责持有并导出原始数据
# ---------------------------------------------------------------------------
@dataclass
class UserClientConfig:
    train_file: str
    validation_file: Optional[str] = None
    text_column: str = "sentence"
    label_column: str = "label"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


class UserClient:
    def __init__(self, config: UserClientConfig):
        self.config = config
        delimiter = "\t" if config.train_file.endswith(".tsv") else ","
        data_files = {"train": config.train_file}
        if config.validation_file is not None:
            data_files["validation"] = config.validation_file

        logger.info("UserClient: loading local dataset %s", data_files)
        self.dataset = load_dataset("csv", data_files=data_files, delimiter=delimiter)

    def _subset(self, split: str, limit: Optional[int]) -> Dict[str, List]:
        data = self.dataset[split]
        if limit is not None:
            limit = min(limit, len(data))
            data = data.select(range(limit))
        return data.to_dict()

    def export_payload(self) -> Dict:
        payload = {
            "text_column": self.config.text_column,
            "label_column": self.config.label_column,
            "splits": {
                "train": self._subset("train", self.config.max_train_samples),
            },
        }
        if "validation" in self.dataset:
            payload["splits"]["validation"] = self._subset(
                "validation", self.config.max_eval_samples
            )
        logger.info("UserClient: payload splits -> %s", list(payload["splits"].keys()))
        return payload


# ---------------------------------------------------------------------------
# 中间网关：将文本转换为 embeddings，并预留噪声添加位置
# ---------------------------------------------------------------------------
@dataclass
class GatewayConfig:
    model_name_or_path: str = "bert-base-uncased"
    max_length: int = 64
    add_noise: bool = False  # 预留，可在此启用 DP 机制


class PrivacyGateway:
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name_or_path)
        self.model.eval()

    @torch.no_grad()
    def _encode_split(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        embeddings = self.model.bert.embeddings(
            input_ids=encodings["input_ids"],
            token_type_ids=encodings.get("token_type_ids"),
        )

        # 预留：此处可以对 embeddings 添加差分隐私噪声
        if self.config.add_noise:
            raise NotImplementedError("Noise injection not implemented in this demo.")

        sanitized = {
            "embeddings": embeddings.cpu(),  # [batch, seq, hidden]
            "attention_mask": encodings["attention_mask"].cpu(),
        }
        return sanitized

    def sanitize(self, payload: Dict) -> Dict:
        text_col = payload["text_column"]
        label_col = payload["label_column"]

        sanitized_payload = {"splits": {}}
        for split_name, split_data in payload["splits"].items():
            texts = split_data[text_col]
            labels = torch.tensor(split_data[label_col], dtype=torch.long)

            sanitized_split = self._encode_split(texts)
            sanitized_split["labels"] = labels
            sanitized_payload["splits"][split_name] = sanitized_split

            logger.info(
                "Gateway: processed split '%s' -> embeddings shape %s",
                split_name,
                tuple(sanitized_split["embeddings"].shape),
            )
        return sanitized_payload


# ---------------------------------------------------------------------------
# 算力服务器：接收 embeddings，继续执行微调
# ---------------------------------------------------------------------------
class EmbeddingDataset(TorchDataset):
    def __init__(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings.float()
        self.attention_mask = attention_mask.long()
        self.labels = labels.long()

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return {
            "inputs_embeds": self.embeddings[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


@dataclass
class ComputeConfig:
    model_name_or_path: str = "bert-base-uncased"
    output_dir: str = "role_embeddings_output"
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    logging_steps: int = 10
    seed: int = 42


class ComputeServer:
    def __init__(self, config: ComputeConfig):
        self.config = config

    def _build_dataset(self, split: Dict[str, torch.Tensor]) -> EmbeddingDataset:
        return EmbeddingDataset(
            embeddings=split["embeddings"],
            attention_mask=split["attention_mask"],
            labels=split["labels"],
        )

    def fine_tune(self, sanitized_payload: Dict) -> Dict:
        set_seed(self.config.seed)

        train_dataset = self._build_dataset(sanitized_payload["splits"]["train"])
        eval_dataset = self._build_dataset(sanitized_payload["splits"]["validation"])

        config = AutoConfig.from_pretrained(
            self.config.model_name_or_path,
            num_labels=len(torch.unique(train_dataset.labels)),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path,
            config=config,
        )

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            num_train_epochs=self.config.num_train_epochs,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="epoch",
            save_strategy="no",
            report_to="none",
            seed=self.config.seed,
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
            eval_dataset=eval_dataset,
            tokenizer=None,  # 直接传入 embeddings，不需要 tokenizer
            data_collator=None,  # 数据集已是张量，默认 collate 足够
            compute_metrics=compute_metrics,
        )

        logger.info("ComputeServer: start fine-tuning with %d samples.", len(train_dataset))
        trainer.train()

        logger.info("ComputeServer: running evaluation.")
        metrics = trainer.evaluate()
        logger.info("Validation metrics: %s", metrics)
        print("Validation metrics:", metrics)
        return metrics


# ---------------------------------------------------------------------------
# 主流程：串联三个角色
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    user = UserClient(
        UserClientConfig(
            train_file="datasets/SST-2/train.tsv",
            validation_file="datasets/SST-2/dev.tsv",
            max_train_samples=32,   # 为演示限制样本量
            max_eval_samples=16,
        )
    )
    raw_payload = user.export_payload()

    gateway = PrivacyGateway(GatewayConfig())
    sanitized_payload = gateway.sanitize(raw_payload)

    server = ComputeServer(ComputeConfig())
    server.fine_tune(sanitized_payload)


if __name__ == "__main__":
    main()
