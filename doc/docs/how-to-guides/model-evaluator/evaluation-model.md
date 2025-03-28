---
sidebar_position: 16
---
# 評価に使うモデルを選ぶ

[自動評価](../../tutorials/evaluation.md)で使うモデルを下記のように指定することができます。

* `<FINETUNED_MODEL_NAME>`：ファインチューン済みのモデル
* `<BASELINE_MODEL_NAME>`：ファインチューン前のモデル
* `<EVALUATOR_MODEL_NAME>`：評価器のモデル（省略可）

評価器のモデルおよび回答の推論用プロンプトを指定しない場合:

```sh
python src/run_model_evaluator.py <FINETUNED_MODEL_NAME> <BASELINE_MODEL_NAME> --data-files data/OpenCL_API.jsonl --output outputs/OpenCL_API_eval.csv
```

評価器のモデルおよび回答の推論用プロンプトを指定する場合:

```sh
python src/run_model_evaluator.py <FINETUNED_MODEL_NAME> <BASELINE_MODEL_NAME> --evaluator <EVALUATOR_MODEL_NAME> --data-files data/OpenCL_API.jsonl --output outputs/OpenCL_API_eval.csv --inference-template templates/<PROMPTFOO_PROMPT_TEMPLATE_NAME_FOR_INFERENCE>.jinja2
```

`<PROMPTFOO_PROMPT_TEMPLATE_NAME_FOR_INFERENCE>` :回答の推論用プロンプトについては[評価に使うプロンプトを選ぶ](evaluation-prompt.md)を参照ください。

一部のモデルは利用するために認証が必要です。[Gated Modelを利用する手順](../etc/huggingface-gated-model.md)を参照してください。
