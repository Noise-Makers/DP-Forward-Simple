# DP-Forward-Simple

## 用法

安装第三方库

```
pip3 install -r ./requirements.txt
```

执行 `python3 ./demo.py` 即可。在 3.8.20 版本的 python 环境中运行通过。默认输出结果保存在 `output` 目录

## 说明

+ 直接加噪的话会极大程度上影响准确性，目前这一块暂时留空。
+ 数据集在 `datasets` 目录中，通过 [GLUE-baselines](https://github.com/nyu-mll/GLUE-baselines) 工具下载，因为不同任务所定义的数据集格式不同，这里展示的是情感分析任务。
+ `demo.py` 中为了简化处理，固定了预训练的模型为 `bert-base-uncased`，后面会考虑其他的模型和任务。
+ 训练过程的参数也基本上固定，如果想要得到更精确的模型，可以将 `num_train_epochs` 的数值给调大一点，比如从`5.0`调到`10.0`。
+ 因为为了演示流程正确性，样本数量也是设置得非常少，这个数值由 `max_train_samples` 和 `max_eval_samples` 确定。如果服务器性能不错，可以将 `max_train_samples` 设置为 `1000` 或以上，`max_eval_samples` 设置为 `200` 或以上。
+ 如果不需要提供验证数据集，那么直接将 `validation_file="datasets/SST-2/dev.tsv",` 注释掉即可，这样就只服务器就只完成训练，而不需要去执行验证。
