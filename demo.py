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
    max_length: int = 64  # 填充长度
    add_noise: bool = False  # 预留，开启则为添加噪声
    dp_parameters: Optional[List] = None  # 预留，加噪相关参数


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
        # 对 payload 转换为 embeddings，然后加噪（加噪部分待实现）
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
                f"[算安保]-处理 {split_name} 数据集 -> embeddings 形式"
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

        # 预留，此处可以对 embeddings 添加噪声
        if self.gate_config.add_noise:
            raise NotImplementedError("尚未提供加噪算法")

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
        eval_dataset = {}
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
            eval_strategy="epoch",
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
            eval_dataset=eval_dataset,
            tokenizer=None,  # 我们直接传入 embeddings，不需要 tokenizer
            data_collator=None,  # 数据集已是张量，默认 collate 足够
            compute_metrics=compute_metrics,
        )

        logger.info(f"[算力网]-开始进行微调，样本规模为 {len(train_dataset)}")
        trainer.train()

        if eval_flag:
            logger.info(f"[算力网]-开始验证")
            metrics = trainer.evaluate()
            print(f"验证结果为：{metrics}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s : %(message)s")
    user_config = UserClientConfig(
        train_file="datasets/SST-2/train.tsv",
        validation_file="datasets/SST-2/dev.tsv",
        model_name="bert-base-uncased",
        max_train_samples=32,   # 为演示限制样本量
        max_eval_samples=16,
    )

    # 设定算安保对象
    gateway = PrivacyGateway(user_config=user_config, gate_config=GatewayConfig())
    raw_payload = gateway.load_data()
    sanitized_payload = gateway.sanitize(raw_payload)
    output_content = gateway.output()  # 这个内容就是输出给算力网的

    # 设定算力服务器对象
    server = ComputeServer(output_content)
    server.fine_tune()
