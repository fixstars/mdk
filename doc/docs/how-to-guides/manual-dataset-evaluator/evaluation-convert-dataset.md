---
sidebar_position: 18
---
# 手動評価結果をデータセットに変換する

このドキュメントでは、手動評価を行った複数の結果を統合し、データセットに変換する手順を説明します。

## 実行方法

次のコマンドで以下の4つのファイルを作成します

```sh
python scripts/merge_dataset.py <TARGET_PATHS> --output-path <OUTPUT_PATH> --output-basename <OUTPUT_BASENAME>
```

* `dataset_<OUTPUT_BASENAME>_checked.csv`（正解ありのデータセット）
* `dataset_<OUTPUT_BASENAME>_checked.xlsx`（正解ありのデータセット）
* `dataset_<OUTPUT_BASENAME>_checked.json`（正解ありのデータセット）
* `dataset_<OUTPUT_BASENAME>.json`（正解なしのデータセット）

## 入出力ファイルの例

* outputs
  * dataset_converter
    * 2024-06-05
      * 1.11-25-06_AI_Business_Guideline
        * experiment_log_manual_2024-06-11-19-16-23.json
      * 2.17-25-57_prtimes_llm
        * experiment_log_manual_2024-06-12-14-05-42.json
      * ...
      * dataset_v2406_checked.csv
      * dataset_v2406_checked.xlsx
      * dataset_v2406_checked.json
      * dataset_v2406.json

### 実行例

`<TARGET_PATHS>`にフォルダを指定した場合は、フォルダ以下に存在する複数フォルダの最新の`experiment_log_manual_*.json`を読み取って入力とします。

```sh
python scripts/merge_dataset.py outputs/dataset_converter/2024-06-05/ --output-path outputs/dataset_converter/2024-06-05/ --output-basename v2406
```

`<TARGET_PATHS>`に複数の`experiment_log_manual_*.json`ファイルを直接指定することもできます。

```sh
python scripts/merge_dataset.py outputs/dataset_converter/2024-06-05/1.11-25-06_AI_Business_Guideline/experiment_log_manual_2024-06-11-19-16-23.json outputs/dataset_converter/2024-06-05/2.17-25-57_prtimes_llm/experiment_log_manual_2024-06-12-14-05-42.json --output-path outputs/dataset_converter/2024-06-05/ --output-basename v2406
```
