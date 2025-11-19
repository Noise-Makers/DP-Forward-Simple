# DP-Forward-Simple

## 指南

安装第三方库

```
pip3 install -r ./requirements.txt
```

执行 `python3 ./demo.py` 即可。默认输出结果保存在 `output` 目录

## 说明

1. 加噪算法采用的是  analytic Gaussian mechanism，参考：[*Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising*](https://proceedings.mlr.press/v80/balle18a/balle18a.pdf)。
2. 差分隐私采用的是 token-level 定义，参考：[*DP-Forward: Fine-tuning and Inference on Language Models with Differential Privacy in Forward Pass*](https://dl.acm.org/doi/10.1145/3576915.3616592)。
3. 数据集在 `datasets` 目录中，通过 [GLUE-baselines](https://github.com/nyu-mll/GLUE-baselines) 工具下载，因为不同任务所定义的数据集格式不同，这里展示的是情感分析任务。
4. `demo.py` 中为了简化处理，固定了预训练的模型为 `bert-base-uncased`，后面会考虑其他的模型和任务。
5. 为了演示流程正确性，样本数量也是设置得非常少，这个数值由 `max_train_samples` 和 `max_eval_samples` 确定。如果服务器性能不错，可以将 `max_train_samples` 设置为 `1000` 或以上，`max_eval_samples` 设置为 `200` 或以上。
6. 如果不需要提供验证数据集，那么直接将 `validation_file="datasets/SST-2/dev.tsv",` 注释掉即可，这样就只服务器就只完成训练，而不需要去执行验证。
7. 直接加噪的话会极大程度上影响准确性，目前还在进行调试。
