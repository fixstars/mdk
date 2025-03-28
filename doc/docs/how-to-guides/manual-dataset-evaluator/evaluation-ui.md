---
sidebar_position: 17
---
# 手動評価UIを利用する

このドキュメントでは、[手動評価のチュートリアル](../../tutorials/evaluation-human-in-the-loop.md)で使用したUIについての各種機能を説明します。

## データセット評価UI

### ページ送り機能

左下の`position`テキストボックスから離れた質問回答に移動できます。そのとき表示されていた質問回答が保存されないことに注意してください。

![manual-dataset-evaluator](../../tutorials/images/manual-dataset-evaluator.png)

### 自動保存機能

10分ごとに作業結果を自動保存します。保存先は入力ファイルと同じディレクトリの`experiment_log_autosave.json`です。

この結果からQAデータセットを作る場合は、一度`src/manual_dataset_evaluator.py`でロードしてから`save`ボタンで保存してください。

## モデル評価UI

モデル評価UIの詳細については[promptfooのドキュメント](https://www.promptfoo.dev/docs/intro)を参照してください。
