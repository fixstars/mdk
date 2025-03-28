---
sidebar_position: 5
---
# DPOモデルの学習

このドキュメントでは、通常のSFTの代わりに[DPO](https://arxiv.org/abs/2305.18290)を使用して学習する方法について説明します。

## データセット構築

DPOの学習には、質問に対する回答2種類と、どちらが優れているかのラベルが必要です。ここでは、[通常のデータセット構築](../../tutorials/step-by-step.md)を行った後`experiment_log.json`に対してさらに変換を実行することでDPOの学習用データセットを構築します。

```terminal
$ python src/run_dataset_converter.py --config-name=dpo inputs.name=<path-to-dataset-convert-result>/experiment_log.json

[2024-04-26 18:23:44,216][preprocess.llm][WARNING] - no examples found. converting without examples...
[2024-04-26 18:23:44,219][preprocess.llm][INFO] - init PreprocessWithLLM: 71.273991 sec
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████| 152/152 [00:34<00:00,  4.38it/s]
[2024-04-26 18:24:19,346][preprocess.dpo_convert][INFO] - dpo_convert: 35.126735 sec / 10 pages = 3.512674 sec/page
[2024-04-26 18:24:19,427][__main__][INFO] - execution time: 106.556117 sec
```

出力ディレクトリには、DPO学習用データセット`dataset.json`と実験ログ`experiment_log.json`が含まれています。

```terminal
$ ls outputs/dataset_converter/2024-04-26/18-22-32
dataset.json  experiment_log.json  run_dataset_converter.log
```

## 学習

`mode=dpo`を指定することでDPOの学習ができます。

```terminal
deepspeed --hostfile=/dev/null src/run_trainer.py config/trainer/Swallow-70b.yaml mode=dpo dataset.path=json dataset.data_files=outputs/dataset_converter/2024-04-26/16-30-14/dataset.json
```

注意事項として、現在`DPOTrainer`のconfigは（`SFTTrainer`との互換性の問題のため）コマンドラインから指定できない状態になっています。`SFTTrainer`の実装を参考にコードを書き換えると、`SFTTrainer`の代わりに`DPOTrainer`のconfigをコマンドラインから指定できる状態にすることができるので、必要に応じて切り替えてください。
