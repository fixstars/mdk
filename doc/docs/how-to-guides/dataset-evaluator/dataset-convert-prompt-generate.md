---
sidebar_position: 16
---
# プロンプトを自動生成する

現在のデータセットとその評価結果を参照して新しいプロンプトを作るのは人手でも可能ですが、`dataset_evaluator`を利用すると自動生成することもできます。

次のコマンドを入力すると、現在使用していたプロンプトテンプレートとその結果作られた質問回答を参考に、新しいプロンプトテンプレートを出力ディレクトリに`template_*.jinja2`というファイルとして出力します。

注意点として、これらが既存のプロンプトテンプレートのフォーマットに沿っているかは厳密にはチェックされていないので、手動での修正が必要になる場合があります。

```sh
python src/run_dataset_evaluator.py --config-name=prompt_generate inputs.name=outputs/dataset_evaluator/<YYYY-MM-DD>/<HH-MM-SS>/experiment_log.json
```

生成されたファイルを利用するには、下記のようにシンボリックリンクを作成します。

```sh
pushd templates && ln -s ../outputs outputs && popd
python src/run_dataset_converter.py inputs.name=data/OpenCL_API_23_32.pdf prompt.template=outputs/dataset_evaluator/<YYYY-MM-DD>/<HH-MM-SS>/template_0.jinja2
```
