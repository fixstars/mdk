---
sidebar_position: 15
---
# 正誤判定結果の自動調整

`karakuri_apm`による自動評価では5段階9属性の評価結果が出力されるので、それを「データセットが正しいか誤りかの二値分類」に変換する必要があります。
しかし、人間が目視で確認し調整することは手間がかかり、また人によるばらつきも生じるため最適とは限りません。
よって自動評価結果（json形式）および、手動評価を行ったデータセット（xlsx形式）から、ロジスティック回帰を用いて自動化しました。
`src/preprocess/json_finder.py`の`raw_score()`関数で用いる評価結果から正誤判定結果を求める係数を算出し、再調整できます。これはその手法を記述したドキュメントです。

* 実行すると、yaml形式の係数ファイルが`<OUT_YAML_PATH>`に作成されます。

## 実行方法

### 実行コマンド

```sh
python scripts/discriminant_analysis.py <BASE_JSON_PATH> <GT_XLSX_PATH> <OUT_YAML_PATH> --seed <SEED>
```

### 実行例

```sh
python scripts/discriminant_analysis.py outputs/dataset_evaluator/2024-07-18_OK/15-54-28_x100/experiment_log.json outputs/plot_data_karakuri_apm/dataset_v2406_checked.xlsx config/dataset_evaluator/coefficient.yaml
```

### 実行結果例

```sh

dump coefficient to config/dataset_evaluator/coefficient.yaml
LogisticRegression() Accuracy: 0.566336 Recall: [0.62838983 0.50281914] Presicion: [0.5640213  0.56932395] F-measure: [0.5944682  0.53400891]
```
