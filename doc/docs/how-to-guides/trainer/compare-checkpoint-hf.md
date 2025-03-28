---
sidebar_position: 11
---
# HuggingFace Transformers形式のチェックポイントの行列を比較する

本説明書では、HuggingFace Transformers形式チェックポイントの行列比較について解説します。

## 前提条件

本手順は以下の構成を前提としています。

* 確認したいモデルに対して[チェックポイントの形式変換(Megatron-LM, HuggingFace Transformers)](./convert-checkpoint-megatron-lm.md)の作業がすべて完了している。
* [データセットを生成してチャットボットを展開するチュートリアル](../../tutorials/step-by-step.md)の「環境の構築」までが完了している。
* MDKのパッケージを展開済みで、`cd <MDKパッケージをcloneした場所>/outputs`を実行し、カレントディレクトリが変更済みである。

## 比較スクリプトの使用方法

比較スクリプト`scripts/compare_weights.py`によって、二つのモデルの`state_dict`を比較することができます。
使い方は以下です。

```sh
python3 compare_weights.py <Reference Checkpoint> <Test Checkpoint>
 # 参照元となるチェックポイントを第一引数に、テスト対象となるチェックポイントを第二引数に指定
```

なお、HuggingFace Transformersライブラリでは、`wrapper`によって自動で`key`の不一致の補完や、定数テンソルの再計算が行われる場合があるため、`state_dict`は完全一致している必要はありません。

## 使用方法の具体的な例

具体的な例として、[チェックポイントの形式変換(Megatron-LM, HuggingFace Transformers)](./convert-checkpoint-megatron-lm.md)で得られたチェックポイントが変換前後で変わっていないことを確認する手順を紹介します。

### Llama2の形式変換確認

行列(`state_dict`)が変換前後で変わっていないことを確認します。
なお、比較できるのは

* HuggingFaceからダウンロードしたLlama-2-7b-hf
* HuggingFaceからダウンロードしたLlama-2-7b-hfをMegatron-LM形式に変換したあと、再度HuggingFace Transformers形式に戻したチェックポイント

です。
HuggingFace Transformers形式のチェックポイントとMegatron-LM形式のチェックポイントの比較ではない点にご注意ください。

```sh
$ . ../.venv/bin/activate
$ python3 ../scripts/compare_weights.py \
    Llama-2-7b-hf \
    Llama-2-7b-mega-hf
Load Llama-2-7b-hf: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:06<00:00,  3.01s/it]
Load Llama-2-7b-mega-hf: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.09s/it]
Compare: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 291/291 [00:37<00:00,  7.75it/s]
Tensors only in the base model
  - model.layers.0.self_attn.rotary_emb.inv_freq
  - model.layers.1.self_attn.rotary_emb.inv_freq

~~(途中省略)~~

  - model.layers.8.self_attn.rotary_emb.inv_freq
  - model.layers.9.self_attn.rotary_emb.inv_freq
Tensors only in the target model
  Nothing
Shape mismatched tensors
  Nothing
Value mismatched tensors
  Nothing

Total tensors: 323
  Tensors only in the base model: 32
  Tensors only in the target model: 0
  Shape mismatched tensors: 0
  Value mismatched tensors: 0
```

上記のように表示されれば、変換は正常に行われています。

#### `self_attn.rotary_emb.inv_freq`について補足

Tensors only in the base modelとして`model.layers.XXXX.self_attn.rotary_emb.inv_freq`が出力されますが、こちらは問題ありません。

`model.layers.XXXX.self_attn.rotary_emb.inv_freq`は、Position Embeddingの層であり、再算出可能なものなのでチェックポイントに含める必要はありません。

### Llama 3の形式変換の確認

Llama 3を比較します。

```sh
$ . ../.venv/bin/activate
$ python3 ../scripts/compare_weights.py \
    Meta-Llama-3-8B \
    Meta-Llama-3-8B-mega-hf
Load Meta-Llama-3-8B: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.01it/s]
Load Meta-Llama-3-8B-mega-hf: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.46s/it]
Compare: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 291/291 [00:40<00:00,  7.20it/s]
Tensors only in the base model
  Nothing
Tensors only in the target model
  Nothing
Shape mismatched tensors
  Nothing
Value mismatched tensors
  Nothing

Total tensors: 291
  Tensors only in the base model: 0
  Tensors only in the target model: 0
  Shape mismatched tensors: 0
  Value mismatched tensors: 0
```

上記のように表示されれば、変換は正常に行われています。

### GPT-2の形式変換の確認

GPT-2を比較します。

```sh
$ . ../.venv/bin/activate
$ python3 ../scripts/compare_weights.py \
    --base-prefix transformer. \
    gpt2-medium \
    gpt2-medium-mega
Load gpt2-medium: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 183.31it/s]
Load gpt2-medium-mega: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.42it/s]
Compare: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 316/316 [00:04<00:00, 75.20it/s]
Tensors only in the base model
  Nothing
Tensors only in the target model
  - lm_head.weight
  - transformer.h.0.attn.masked_bias
  - transformer.h.1.attn.masked_bias

~~(途中省略)~~

  - transformer.h.8.attn.masked_bias
  - transformer.h.9.attn.masked_bias
Shape mismatched tensors
  - transformer.wte.weight
Value mismatched tensors
  Nothing

Total tensors: 341
  Tensors only in the base model: 0
  Tensors only in the target model: 25
  Shape mismatched tensors: 1
  Value mismatched tensors: 0
```

上記のように表示されれば、変換は正常に行われています。

#### `transformer.wte.weight`について補足

Shape mismatched tensorsとして`transformer.wte.weight`が出力されますが、こちらは問題ありません。

HuggingFace Transformers形式からMegatron-LM形式へ変換する際に、`Vocab size`が変わったことが原因です。（いくつかの`dummy token`が追加されました）

#### `lm_head.weight`について補足

Tensors only in the target modelとして`lm_head.weight`が出力されますが、こちらは問題ありません。

これはMegatron-LM形式からHuggingFace Transformers形式へ変換する際に追加されるTensorで、元のモデルに含まれている必要はありません。

#### `attn.masked_bias`について補足

Tensors only in the target modelとして`transformer.h.XXXX.attn.masked_bias`が出力されますが、こちらは問題ありません。

これらはMegatron-LM形式からHuggingFace Transformers形式へ変換する際に追加されるダミーTensorで、元のモデルに含まれている必要はありません。

#### `state_dict`の`key`の差分について補足

`/openai-community/gpt2-medium`モデルの`state_dict`の`key`は以下のようになっています。

```json
[
        "h.0.ln_1.weight",
        "h.0.ln_1.bias",
        "h.0.attn.bias",
        "h.0.attn.c_attn.weight",
        "h.0.attn.c_attn.bias",
        "h.0.attn.c_proj.weight",
        "h.0.attn.c_proj.bias",
        "h.0.ln_2.weight",
        "h.0.ln_2.bias",
        "h.0.mlp.c_fc.weight",
        "h.0.mlp.c_fc.bias",
        "h.0.mlp.c_proj.weight",
        "h.0.mlp.c_proj.bias",
        ...
]
```

一方で、Megatron-LM形式から、HuggingFace Transformers形式への変換後は以下のような`key`で出力されます。違いは接頭語`transformer.`の有無です。

```json
[
        "transformer.h.0.ln_1.weight",
        "transformer.h.0.ln_1.bias",
        "transformer.h.0.attn.bias",
        "transformer.h.0.attn.masked_bias",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_attn.bias",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.attn.c_proj.bias",
        "transformer.h.0.ln_2.weight",
        "transformer.h.0.ln_2.bias",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_fc.bias",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.0.mlp.c_proj.bias",
        ...
]
```

前者を「フォーマットA」、後者を「フォーマットB」とします。

transformersにある変換スクリプトでは、フォーマットBにのみ対応しています。
そのため[チェックポイントの形式変換(Megatron-LM, HuggingFace Transformers)](./convert-checkpoint-megatron-lm.md)の「HuggingFace Transformers形式からMegatron-LM形式へ変換する」では、修正スクリプト(`scripts/pretrain_megatron_lm/convert_checkpoint/fix_statedict_key_prefix_gpt2.py`)を適用し、`/openai-community/gpt2-medium`モデル（フォーマットA）をフォーマットBに変換しています。

比較の際は`--base-prefix transformer.`の指定を追加することで、比較元の`key`をフォーマットB相当にしたうえで比較しています。
