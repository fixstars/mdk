---
sidebar_position: 1
---
# モデルの開発を始める

MDK (Model Development Kit)は、ドキュメントに関する指示応答ができるようにファインチューンされたLLMを開発するためのシステムです。ユーザーはMDKで作ったLLMを利用して独自のアプリケーションを簡単に構築することができます。

## 要件

MDKを動作させるには、GPUが搭載されたサーバーが必要です。そのうえで、一般的なAI学習環境は構築済みであることが想定されています。

そのような環境として、[Fixstars AI Booster](https://www.fixstars.com/ja/ai/booster) (FAIB)の利用が推奨されています。
以下のいくつかの説明でも、FAIB環境・サーバーを前提としている場合がありますので、MDKを快適かつ高速に実行するためにも、FAIBのご利用を検討ください。

## 解説

LLMアプリケーションを開発するために一般的に求められる知識についての解説記事です。LLMアプリケーションに初めて触れる方は、ご一読いただくと以降のチュートリアル等が進めやすくなります。

* [LLMアプリケーションの作り方](explanation/llmops.md)

## チュートリアル

上から順番に実行するだけでMDKの主要な機能すべてを試すことができるチュートリアルを用意しました。はじめて利用する方はこちらからお試しください。

* [データセットを生成してチャットボットを展開する](tutorials/step-by-step.md)
* [データセットとモデルを評価する](tutorials/evaluation.md)
* [人力評価してフィードバックする](tutorials/evaluation-human-in-the-loop.md)
* [パイプラインを自動実行する](tutorials/auto-pipeline.md)

## ハウツー

より良いモデルを作るためには、各種パラメータやプロンプトの調整が欠かせません。MDKでは調整する対象がモジュールごとに分かれており、下記のドキュメントに沿って簡単に調整できるように設計されています。

モジュール名と機能の対応については、上記の解説もしくはチュートリアルを参照ください。

* `dataset_converter`
  * [データセット変換器の設定を変更する](how-to-guides/dataset-converter/dataset-convert-adjust.md)
  * [データセットへの変換例](how-to-guides/dataset-converter/dataset-convert-examples.md)
* `trainer`
  * [学習対象のモデルを選ぶ](how-to-guides/trainer/train-model.md)
  * [MLflowを使う](how-to-guides/trainer/use-mlflow.md)
  * [DPOを使う](how-to-guides/trainer/train-dpo.md)
  * [Megatron-LMを用いた分散並列事前学習](how-to-guides/trainer/pretrain-megatron-lm.md)
  * [Megatron-LMの学習設定を変更する](how-to-guides/trainer/pretrain-megatron-lm-settings.md)
  * [チェックポイントの形式を変換する（Megatron-LM形式\<\-\> HuggingFace Transformers形式）](how-to-guides/trainer/convert-checkpoint-megatron-lm.md)
  * [HuggingFace Transformers形式のチェックポイントの行列を比較する](how-to-guides/trainer/compare-checkpoint-hf.md)
* `dataset_evaluator`
  * [評価指標を設定する](how-to-guides/dataset-evaluator/evaluation-metrics.md)
  * [正誤判定結果の自動調整](how-to-guides/dataset-evaluator/evaluation-calculate-coefficient.md)
  * [プロンプトを自動生成する](how-to-guides/dataset-evaluator/dataset-convert-prompt-generate.md)
* `model_evaluator`
  * [評価に使うモデルを選ぶ](how-to-guides/model-evaluator/evaluation-model.md)
  * [評価に使うプロンプトを選ぶ](how-to-guides/model-evaluator/evaluation-prompt.md)
* `manual_dataset_evaluator`
  * [手動評価UIを利用する](how-to-guides/manual-dataset-evaluator/evaluation-ui.md)
  * [手動評価結果をデータセットに変換する](how-to-guides/manual-dataset-evaluator/evaluation-convert-dataset.md)
* `launcher`
  * [チャットの詳細設定を変更する](how-to-guides/launcher/launch-settings.md)
  * [VSCode拡張でモデルを利用する](how-to-guides/launcher/vscode-integration.md)
* `pipeline`
  * [自動パイプラインの使い方](how-to-guides/pipeline/pipeline-usage.md)
* その他
  * [Gated modelの利用方法](how-to-guides/etc/huggingface-gated-model.md)
  * [CUDA out of memoryエラーに対応する](how-to-guides/etc/out-of-memory.md)
