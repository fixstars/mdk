---
sidebar_position: 16
---
# 評価に使うプロンプトを選ぶ

自動評価の結果を改善するため、いくつかのプロンプトが用意されています。評価に使うモデルに合わせて、プロンプトを選択したり書き直したりしてください。

## データセット評価

### `verify_dataset_find_error.jinja2`

質問回答の組とその元になったデータが入力され、その誤りの個数を出力するように指示します。

## モデル評価

### 回答の推論用プロンプト

[自動評価](../../tutorials/evaluation.md)で使う回答の推論用プロンプト`<PROMPTFOO_PROMPT_TEMPLATE_NAME_FOR_INFERENCE>`を下記のように指定することができます。（省略可）

```sh
python src/run_model_evaluator.py <FINETUNED_MODEL_NAME> <BASELINE_MODEL_NAME> --evaluator <EVALUATOR_MODEL_NAME> --data-files data/OpenCL_API.jsonl --output outputs/OpenCL_API_eval.csv --inference-template templates/<PROMPTFOO_PROMPT_TEMPLATE_NAME_FOR_INFERENCE>.jinja2
```

#### `promptfoo_default.jinja2`

質問だけを入力し、LLMに回答を答えさせます。`--inference-template`引数省略時の、デフォルトのプロンプトテンプレートです。

#### `promptfoo_swallow_70b.jinja2`

用途は`promptfoo_default.jinja2`と同じですが、Swallow 70B向けのシステムプロンプトテンプレートです。

### ルーブリックプロンプト

#### `config/model_evaluator/promptfooconfig.yaml`

質問に対するLLMの出力と正解を入力し、誤りがあるかないかの二値分類を実行します。

#### `config/model_evaluator/promptfooconfig_rubric_jp.yaml`

中身は`config/model_evaluator/promptfooconfig.yaml`と同じですが、日本語で書かれています。評価結果が日本語で返ってくることが期待できます。
