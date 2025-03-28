---
sidebar_position: 22
---
# 自動パイプラインの使い方

`run_pipeline.py`に直接指定して設定できる引数は以下のとおりです。

* 学習: `--model-cfg`
* モデル評価: `--baseline`, `--evaluator`

その他のモジュールの設定を変更したい場合は、`config/`以下のファイルを直接編集してください。

## 実行の再開

一度完了または中断したパイプラインを、`--resume-model`と`--resume-data`引数を使って再開させることができます。

```sh
python src/run_pipeline.py data/OpenCL_API.pdf --resume-model <PATH_TO_MODEL>/checkpoint-<STEP>-merged --resume-data data/OpenCL_API.jsonl
```

ただし、`--resume-model`で指定するモデルは`src/model_evaluator/merge_peft_model.py`で変換済みのものを指定してください。そうしないと、継続学習にならず前の学習結果を初期化してしまいます。

## 個別実行

実行ログには、個別に実行するためのコマンドも表示されています。データセットの生成・学習・評価をそれぞれ実行したい場合にそのまま使うことができます。

```terminal
The pipeline started.
pipeline #0 | dataset_converter
python src/run_dataset_converter.py inputs.name=data/OpenCL_API_23_32.pdf hydra.run.dir=outputs/pipeline/20240328_182549/0_dataset_converter seed=0

pipeline #0 | dataset_evaluator
python src/run_dataset_evaluator.py inputs.name=outputs/pipeline/20240328_182549/0_dataset_converter/experiment_log.json hydra.run.dir=outputs/pipeline/20240328_182549/0_dataset_evaluator seed=0

pipeline #0 | trainer
deepspeed --hostfile=/dev/null src/run_trainer.py config/trainer/Swallow-70b.yaml dataset.data_files=outputs/pipeline/20240328_182549/0_dataset_evaluator/dataset_clean_merged.jsonl training_args.output_dir=outputs/pipeline/20240328_182549/0_trainer model.pretrained_model_name_or_path=tokyotech-llm/Swallow-70b-instruct-hf
```
