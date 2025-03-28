---
sidebar_position: 8
---
# Megatron-LMの学習設定を変更する

このドキュメントでは、[Megatron-LMを用いた分散並列事前学習](./pretrain-megatron-lm.md)における学習設定の変更方法について説明します。

## 学習タスクごとに使用GPUを変更する

計算ノード内の8GPUを4GPUずつに分割して、異なる学習タスクを同時に実行したい場合は以下の3つを変更してください。

1. MPIプロセス数（`-np`で指定する数）を使用したいGPU数に合わせる
2. タスク間で使用するGPUが衝突しないように、[CUDA out of memoryエラーに対応するハウツー](../../how-to-guides/etc/out-of-memory.md)に従って環境変数`CUDA_VISIBLE_DEVICES`を設定する
3. タスク間で通信ポートが衝突しないように、学習スクリプト中の`export MASTER_PORT=<定数>`の行をコメントアウトして、環境変数`MASTER_PORT`を個別に設定する

例えば、4GPUごとに分割する場合の実行コマンドは以下のようになります。

```sh
# 学習タスク1
$ mpirun -np 4 -x CUDA_VISIBLE_DEVICES=0,1,2,3 -x MASTER_PORT=50001 ...
# 学習タスク2
$ mpirun -np 4 -x CUDA_VISIBLE_DEVICES=4,5,6,7 -x MASTER_PORT=50002 ...
```

## 学習イテレーションを変更する

`--train-iters`を変更することで、学習したいイテレーション数を変更することができます。
