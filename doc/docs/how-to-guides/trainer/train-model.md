---
sidebar_position: 3
---
# 学習対象のモデルを選ぶ

## DeepSpeedを用いたファインチューン対象のモデルを選ぶ

config/trainerの名前を変えることで、事前学習済みのモデルを選ぶことができます。

```sh
# 0.35B
deepspeed --hostfile=/dev/null src/run_trainer.py config/trainer/opt-350m.yaml dataset.data_files=data/OpenCL_API.jsonl
# 7B
deepspeed --hostfile=/dev/null src/run_trainer.py config/trainer/Xwin-LM-7B-V0.2.yaml dataset.data_files=data/OpenCL_API.jsonl
# 34B
deepspeed --hostfile=/dev/null src/run_trainer.py config/trainer/nontoxic-bagel-34b-v0.2.yaml dataset.data_files=data/OpenCL_API.jsonl
# 8x7b
deepspeed --hostfile=/dev/null src/run_trainer.py config/trainer/mixtral-8x7b.yaml dataset.data_files=data/OpenCL_API.jsonl
# 70B
deepspeed --hostfile=/dev/null src/run_trainer.py config/trainer/Swallow-70b.yaml dataset.data_files=data/OpenCL_API.jsonl
```

一部のモデルは利用するために認証が必要です。[Gated Modelを利用する手順](../etc/huggingface-gated-model.md)を参照してください。

## LoRA対象のレイヤーを変更する

既定の設定では、LoRAは、適用可能なすべてのレイヤーに設定されます。

```terminal
$ deepspeed --hostfile=/dev/null src/run_trainer.py config/trainer/Swallow-70b.yaml dataset.data_files=data/OpenCL_API.jsonl

INFO:__main__:set lora layer to all linear layers ['down_proj', 'gate_proj', 'k_proj', 'lm_head', 'o_proj', 'q_proj', 'up_proj', 'v_proj']

trainable params: 3,326,650,368 || all params: 72,486,406,144 || trainable%: 4.589343774874623
```

もし、特定のレイヤーにだけLoRAを適用したい場合、`lora_config.target_modules`引数で指定してください。指定可能な名前は標準的にはモデル中の`nn.Linear`レイヤのattribute nameで、その一覧は上記の`['down_proj', 'gate_proj', 'k_proj', 'lm_head', 'o_proj', 'q_proj', 'up_proj', 'v_proj']`のように通常実行時に表示されています。

```terminal
$ deepspeed --hostfile=/dev/null src/run_trainer.py config/trainer/Swallow-70b.yaml dataset.data_files=data/OpenCL_API.jsonl lora_config.target_modules=[q_proj,k_proj,v_proj,o_proj]

trainable params: 1,048,576,000 || all params: 70,208,331,776 || trainable%: 1.4935207452948553
```

## Megatron-LMを用いた学習対象のモデルを選ぶ

Megatron-LMを用いた学習について、いくつかのモデルでの学習方法を示します。
なお、以下の作業が完了していることを前提としています。

* [Megatron-LMを用いた分散並列事前学習](./pretrain-megatron-lm.md)の「データセットの前処理」までの設定が完了している。
* 確認したいモデルについて[チェックポイントの形式変換(Megatron-LM, HuggingFace Transformers)](./convert-checkpoint-megatron-lm.md)の手順に基づき、HuggingFace Transformers形式のチェックポイントをMegatron-LM形式に変換済み。
* `cd <MDKパッケージをcloneした場所>/outputs/Megatron-LM`を実行し、カレントディレクトリを変更済み。

### モデルサイズの変更

[Megatron-LMを用いた分散並列事前学習ハウツー](./pretrain-megatron-lm.md)では、GPT-3 7Bを例として説明しています。

その他のモデルサイズで実験するには複数のパラメータを同時に変更する必要があります。具体的には下表を目安に調整してください。

|パラメータ数|345M|857M|1.7B|7.1B|16B|32B|70B|
|:--|--:|--:|--:|--:|--:|--:|--:|
|`--num-layers`|12|24|24|30|32|48|84|
|`--hidden-size`|512|1024|2048|4096|6144|7168|8192|
|`--num-attention-heads`|8|16|16|32|48|56|64|
|`--seq-length`|1024|2048|2048|2048|2048|2048|1024|
|`--tensor-model-parallel-size`|1|1|1|2|4|8|8|
|`--pipeline-model-parallel-size`|1|1|1|1|1|1|2|

* 参考
  * [https://github.com/NVIDIA/Megatron-LM/tree/main/examples/gpt3](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/gpt3)
    * 345M, 857Mの各項目
    * 他モデルのseq-length（175Bモデルがseq-length=2048であることから類推）
  * [https://github.com/NVIDIA/Megatron-LM/blob/main/images/model_table.png](https://github.com/NVIDIA/Megatron-LM/blob/main/images/model_table.png)
    * 上記を除く各項目

例えば以下のコマンドで、2ノードを用いて70Bモデルの学習を実行することができます。

```sh
cp ../../scripts/pretrain_megatron_lm/train_gpt3_70b_mpi.sh examples/gpt3/
mkdir cache

# FAIBサーバーでは、`/opt/faib/etc/hostfile.txt`を利用することができます。
mpirun --np 16 --hostfile <2ノードのIPアドレスが設定されたhostfile.txt> \
    /opt/singularity/bin/singularity exec --nv --no-home --bind ./cache:$HOME/.triton/cache ./pytorch.sif \
    bash examples/gpt3/train_gpt3_70b_mpi.sh \
        ./checkpoint_gpt3_70b ./tensorboard_gpt3_70b ./gpt2/vocab.json ./gpt2/merges.txt ./arxiv_text_document
```

### モデルアーキテクチャの変更

GPT-3以外のモデル(GPT-2, Llama 2, Llama3)の学習手順を紹介します。

まずキャッシュ書き込み用のフォルダを作成します。

```sh
mkdir cache
```

Llama 2, Llama3の学習を行う場合はデータセットの前処理を行います。

```sh
# Llama 2
singularity exec --nv  --bind ../:/outputs ./pytorch.sif   \
    python  tools/preprocess_data.py \
    --input  arxiv.jsonl \
    --output-prefix llama2_arxiv \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model /outputs/Llama-2-7b-hf/tokenizer.model \
    --workers 64 \
    --append-eod


# Llama 3
singularity exec --env TIKTOKEN_CACHE_DIR="" --nv  --bind ../:/outputs ./pytorch.sif   \
    python  tools/preprocess_data.py \
    --input  arxiv.jsonl \
    --output-prefix llama3_arxiv \
    --tokenizer-type Llama3Tokenizer \
    --tokenizer-model /outputs/Meta-Llama-3-8B/original/tokenizer.model \
    --workers 64 \
    --append-eod

```

次に、学習用スクリプトを用意します。

```sh
# GPT-2 Medium (355M)
mkdir -p examples/gpt2
cp ../../scripts/pretrain_megatron_lm/train_gpt2_medium_mpi.sh examples/gpt2/

# Llama 2 7B
mkdir -p examples/llama2
cp ../../scripts/pretrain_megatron_lm/train_llama2_7b_mpi.sh examples/llama2/

# Llama 3 8B
mkdir -p examples/llama3
cp ../../scripts/pretrain_megatron_lm/train_llama3_8b_mpi.sh examples/llama3/


# Llama 3 70B
mkdir -p examples/llama3
cp ../../scripts/pretrain_megatron_lm/train_llama3_70b_mpi.sh examples/llama3/

```

ランダム初期値から事前学習を行う場合は、コピー後のスクリプト中の`--load $CHECKPOINT_LOAD_PATH`および`--use-checkpoint-args`をコメントアウトします。
変換した重みから継続事前学習・ファインチューニングする場合は不要）

学習を実行します。

```sh
# GPT-2 Medium (355M)
mpirun --np 8 \
    singularity exec --no-home --nv --bind ./cache:$HOME/.triton/cache --bind ../:/outputs ./pytorch.sif \
    bash examples/gpt2/train_gpt2_medium_mpi.sh \
        /outputs/gpt2-medium-mega-tp1-pp1 \
        /outputs/gpt2-medium

# Llama 2 7B
mpirun --np 8 \
    singularity exec --no-home --nv --bind ./cache:$HOME/.triton/cache --bind ../:/outputs ./pytorch.sif \
    bash examples/llama2/train_llama2_7b_mpi.sh \
        /outputs/Llama-2-7b-mega-tp1-pp1 \
        /outputs/Llama-2-7b-hf

# Llama 3 8B
mpirun --np 8 \
    singularity exec --env TIKTOKEN_CACHE_DIR="" --no-home --nv --bind ./cache:$HOME/.triton/cache --bind ../:/outputs ./pytorch.sif \
    bash examples/llama3/train_llama3_8b_mpi.sh \
        /outputs/Meta-Llama-3-8B-mega-tp1-pp1 \
        /outputs/Meta-Llama-3-8B/original

# Llama 3 70B
#   FAIBサーバーでは、`/opt/faib/etc/hostfile.txt`を利用することができます。
mpirun --np 24 --hostfile <3ノードのIPアドレスが設定されたhostfile.txt> \
    /opt/singularity/bin/singularity exec --no-home --nv --bind ./cache:$HOME/.triton/cache --bind ../:/outputs ./pytorch.sif \
    bash examples/llama3/train_llama3_70b_mpi.sh \
        /outputs/Meta-Llama-3-70B-mega-tp8-pp3 \
        /outputs/Meta-Llama-3-70B/

# Llama 3.1 8B
mpirun --np 8 \
    singularity exec --env TIKTOKEN_CACHE_DIR="" --no-home --nv --bind ./cache:$HOME/.triton/cache --bind ../:/outputs ./pytorch.sif \
    bash examples/llama3/train_llama3_8b_mpi.sh \
        /outputs/Llama-3.1-8B-mega-tp1-pp1 \
        /outputs/Llama-3.1-8B/original

# Swallow Llama 3 8B
mpirun --np 8 \
    singularity exec --env TIKTOKEN_CACHE_DIR="" --no-home --nv --bind ./cache:$HOME/.triton/cache --bind ../:/outputs ./pytorch.sif \
    bash examples/llama3/train_llama3_8b_mpi.sh \
        /outputs/Llama-3-Swallow-8B-Instruct-v0.1-mega-tp1-pp1 \
        /outputs/Meta-Llama-3-8B/original

```

学習結果は`CHECKPOINT_SAVE_PATH`で指定したディレクトリに保存されます。

### その他のモデル

モデルやアーキテクチャのサポート情報については[Megatron-LMのリリースノート](https://github.com/NVIDIA/Megatron-LM/releases)を参照してください。
