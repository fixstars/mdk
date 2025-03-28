---
sidebar_position: 2
---
# データセットへの変換例

MDKへ入力できるファイル形式は以下のものがあります。

* `.pdf`
* `.txt`
* `.md`
* [pandoc](https://pandoc.org/)が扱えるもの
  * `.csv`
  * `.html`
  * `.ipynb`
  * `.json`
  * `.tex`
  * `.rst`
  * `.docx`
* 画像
  * [PILのサポートしている画像フォーマット](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)

すべての形式について、`src/run_dataset_converter.py`の`inputs.name`にファイル名またはディレクトリ名を指定することで変換を実行することができます。`inputs.name`にascii以外の文字が含まれるときは`\"`でファイル名を囲む必要があることに注意してください。

以下に、いくつかの形式での変換例を掲載します。

## PDF形式(OpenCL API)

```terminal
$ curl -o data/OpenCL_API.pdf https://registry.khronos.org/OpenCL/specs/3.0-unified/pdf/OpenCL_API.pdf
$ python scripts/cut_pdf_file.py data/OpenCL_API.pdf 23 32
$ python src/run_dataset_converter.py inputs.name=data/OpenCL_API_23_32.pdf

split pdf page: 100%|████████████████████████████████████████████████████████████████████████| 10/10 [00:15<00:00,  1.50s/it]
[2024-07-12 10:09:09,198][preprocess.llm][INFO] - init PreprocessWithLLM: 61.331638 sec
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████| 10/10 [00:28<00:00,  2.89s/it]
[2024-07-12 10:09:38,550][preprocess.llm][INFO] - llm: 1417.434547 token/sec (41018 token / 28.938197 sec) with {'provider': 'vllm', 'model': 'tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1', 'max_tokens': 4096, 'generate': {'num_return_sequences': 4, 'max_new_tokens': 1024, 'temperature': 1, 'top_p': 0.95, 'do_sample': True}, 'batch_size': 16, 'api_base_url': 'http://localhost:8001/v1', 'api_key': 'hoge', 'vllm_tensor_parallel_size': 2, 'vllm_max_num_seqs': 256}
[2024-07-12 10:09:38,557][preprocess.sft_convert][INFO] - convert: 29.358645 sec / 10 pages = 2.935865 sec/page
[2024-07-12 10:09:38,643][preprocess.sft_convert][INFO] - output_dataset: outputs/dataset_converter/2024-07-12/10-07-52/dataset.jsonl
[2024-07-12 10:09:38,645][preprocess.sft_convert][INFO] - output_experimental_log: outputs/dataset_converter/2024-07-12/10-07-52/experiment_log.json
[2024-07-12 10:09:38,645][__main__][INFO] - execution time: 106.561182 sec
```

## Markdown形式(cpprefjp)

```terminal
$ pushd data && curl https://codeload.github.com/cpprefjp/site/tar.gz/master | tar -xz site-master && popd
$ python src/run_dataset_converter.py inputs.name=data/site-master/reference/algorithm/sort.md

[2024-07-12 09:43:59,079][preprocess.llm][INFO] - init PreprocessWithLLM: 63.385078 sec
Processed prompts: 100%|███████████████████████████████████████████████████████████████████████| 5/5 [00:23<00:00,  4.74s/it]
[2024-07-12 09:44:23,389][preprocess.llm][INFO] - llm: 863.831802 token/sec (20509 token / 23.741890 sec) with {'provider': 'vllm', 'model': 'tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1', 'max_tokens': 4096, 'generate': {'num_return_sequences': 4, 'max_new_tokens': 1024, 'temperature': 1, 'top_p': 0.95, 'do_sample': True}, 'batch_size': 16, 'api_base_url': 'http://localhost:8001/v1', 'api_key': 'hoge', 'vllm_tensor_parallel_size': 2, 'vllm_max_num_seqs': 256}
[2024-07-12 09:44:23,394][preprocess.sft_convert][INFO] - convert: 24.314574 sec / 5 pages = 4.862915 sec/page
[2024-07-12 09:44:23,474][preprocess.sft_convert][INFO] - output_dataset: outputs/dataset_converter/2024-07-12/09-42-55/dataset.jsonl
[2024-07-12 09:44:23,475][preprocess.sft_convert][INFO] - output_experimental_log: outputs/dataset_converter/2024-07-12/09-42-55/experiment_log.json
[2024-07-12 09:44:23,475][__main__][INFO] - execution time: 87.782835 sec
```

## Markdownの入ったディレクトリ(cpprefjp)

```terminal
$ pushd data && curl https://codeload.github.com/cpprefjp/site/tar.gz/master | tar -xz site-master && popd
$ python src/run_dataset_converter.py inputs.name=data/site-master/module/

[2024-07-12 09:47:15,045][preprocess.llm][INFO] - init PreprocessWithLLM: 60.864344 sec
(RayWorkerVllm pid=2960902) INFO 07-12 09:47:14 model_runner.py:867] Graph capturing finished in 6 secs.
Processed prompts: 100%|███████████████████████████████████████████████████████████████████████| 4/4 [00:22<00:00,  5.71s/it]
[2024-07-12 09:47:38,786][preprocess.llm][INFO] - llm: 717.206186 token/sec (16407 token / 22.876267 sec) with {'provider': 'vllm', 'model': 'tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1', 'max_tokens': 4096, 'generate': {'num_return_sequences': 4, 'max_new_tokens': 1024, 'temperature': 1, 'top_p': 0.95, 'do_sample': True}, 'batch_size': 16, 'api_base_url': 'http://localhost:8001/v1', 'api_key': 'hoge', 'vllm_tensor_parallel_size': 2, 'vllm_max_num_seqs': 256}
[2024-07-12 09:47:38,791][preprocess.sft_convert][INFO] - convert: 23.744875 sec / 4 pages = 5.936219 sec/page
[2024-07-12 09:47:38,873][preprocess.sft_convert][INFO] - output_dataset: outputs/dataset_converter/2024-07-12/09-46-14/dataset.jsonl
[2024-07-12 09:47:38,874][preprocess.sft_convert][INFO] - output_experimental_log: outputs/dataset_converter/2024-07-12/09-46-14/experiment_log.json
[2024-07-12 09:47:38,874][__main__][INFO] - execution time: 84.695251 sec
```

## docx形式(security rule)

```terminal
$ curl -o data/ipa_security_rule.docx https://www.ipa.go.jp/security/sme/ps6vr7000001bu8m-att/000055794.docx
$ python src/run_dataset_converter.py inputs.name=data/ipa_security_rule.docx

[2024-07-12 09:50:42,747][preprocess.llm][INFO] - init PreprocessWithLLM: 60.957582 sec
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████| 20/20 [00:43<00:00,  2.19s/it]
[2024-07-12 09:51:26,934][preprocess.llm][INFO] - llm: 1839.084737 token/sec (80487 token / 43.764704 sec) with {'provider': 'vllm', 'model': 'tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1', 'max_tokens': 4096, 'generate': {'num_return_sequences': 4, 'max_new_tokens': 1024, 'temperature': 1, 'top_p': 0.95, 'do_sample': True}, 'batch_size': 16, 'api_base_url': 'http://localhost:8001/v1', 'api_key': 'hoge', 'vllm_tensor_parallel_size': 2, 'vllm_max_num_seqs': 256}
[2024-07-12 09:51:26,942][preprocess.sft_convert][INFO] - convert: 44.194300 sec / 20 pages = 2.209715 sec/page
[2024-07-12 09:51:27,028][preprocess.sft_convert][INFO] - output_dataset: outputs/dataset_converter/2024-07-12/09-49-40/dataset.jsonl
[2024-07-12 09:51:27,031][preprocess.sft_convert][INFO] - output_experimental_log: outputs/dataset_converter/2024-07-12/09-49-40/experiment_log.json
[2024-07-12 09:51:27,031][__main__][INFO] - execution time: 106.180439 sec
```

## ipynbの入ったディレクトリ(fixstars-amplify)

```terminal
$ pushd data && curl https://codeload.github.com/fixstars/amplify-examples/tar.gz/main | tar -xz amplify-examples-main && popd
$ python src/run_dataset_converter.py inputs.name=data/amplify-examples-main/notebooks/ja/tutorials

[2024-07-12 10:02:07,963][preprocess.llm][WARNING] - no examples found. converting without examples...
[2024-07-12 10:02:07,966][preprocess.llm][INFO] - init PreprocessWithLLM: 60.205169 sec
(RayWorkerVllm pid=2992314) INFO 07-12 10:02:07 model_runner.py:867] Graph capturing finished in 6 secs.
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:34<00:00,  2.19s/it]
[2024-07-12 10:02:43,577][preprocess.llm][INFO] - llm: 1816.186031 token/sec (63614 token / 35.026148 sec) with {'provider': 'vllm', 'model': 'tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1', 'max_tokens': 4096, 'generate': {'num_return_sequences': 4, 'max_new_tokens': 1024, 'temperature': 1, 'top_p': 0.95, 'do_sample': True}, 'batch_size': 16, 'api_base_url': 'http://localhost:8001/v1', 'api_key': 'hoge', 'vllm_tensor_parallel_size': 2, 'vllm_max_num_seqs': 256}
[2024-07-12 10:02:43,583][preprocess.sft_convert][INFO] - convert: 35.617344 sec / 16 pages = 2.226084 sec/page
[2024-07-12 10:02:43,668][preprocess.sft_convert][INFO] - output_dataset: outputs/dataset_converter/2024-07-12/10-00-53/dataset.jsonl
[2024-07-12 10:02:43,671][preprocess.sft_convert][INFO] - output_experimental_log: outputs/dataset_converter/2024-07-12/10-00-53/experiment_log.json
[2024-07-12 10:02:43,671][__main__][INFO] - execution time: 110.046625 sec
```

## 画像（火山）

```terminal
$ wget -O data/火山.jpg https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg
$ python src/run_dataset_converter.py inputs.name=\"data/火山.jpg\"

[2024-07-12 10:04:52,637][preprocess.llm][INFO] - init PreprocessWithLLM: 60.037776 sec
Processed prompts: 100%|███████████████████████████████████████████████████████████████████████| 1/1 [00:17<00:00, 17.00s/it]
[2024-07-12 10:05:10,134][preprocess.llm][INFO] - llm: 241.277817 token/sec (4103 token / 17.005293 sec) with {'provider': 'vllm', 'model': 'tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1', 'max_tokens': 4096, 'generate': {'num_return_sequences': 4, 'max_new_tokens': 1024, 'temperature': 1, 'top_p': 0.95, 'do_sample': True}, 'batch_size': 16, 'api_base_url': 'http://localhost:8001/v1', 'api_key': 'hoge', 'vllm_tensor_parallel_size': 2, 'vllm_max_num_seqs': 256}
[2024-07-12 10:05:10,139][preprocess.sft_convert][INFO] - convert: 17.502014 sec / 1 pages = 17.502014 sec/page
[2024-07-12 10:05:10,245][preprocess.sft_convert][INFO] - output_dataset: outputs/dataset_converter/2024-07-12/10-03-38/dataset.jsonl
[2024-07-12 10:05:10,245][preprocess.sft_convert][INFO] - output_experimental_log: outputs/dataset_converter/2024-07-12/10-03-38/experiment_log.json
[2024-07-12 10:05:10,245][__main__][INFO] - execution time: 91.322095 sec
```
