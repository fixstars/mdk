#!/bin/bash

# Runs the "70B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

GPUS_PER_NODE=8
# Change for multinode config
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=14535 # 適当な値
export LOCAL_RANK=$((OMPI_COMM_WORLD_RANK % GPUS_PER_NODE))

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32
SEQUENCE_LENGTH=1024

# 継続事前学習を行う場合は以下の並列数を変更すると再度新しい並列数でモデル変換が必要となります
TP=8  # --tensor-model-parallel-size
PP=2  # --pipeline-model-parallel-size

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$5 #<Specify path and file prefix>_text_document

# 70B
GPT_MODEL_ARGS=(
    --num-layers 84
    --hidden-size 8192
    --num-attention-heads 64
    --seq-length $SEQUENCE_LENGTH
    --max-position-embeddings $SEQUENCE_LENGTH
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 8177 # 261664 / 32
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 430000
    --transformer-impl transformer_engine
    # compute method
    --swiglu
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --disable-bias-linear
    --no-masked-softmax-fusion
    --untie-embeddings-and-output-weights
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
    --sequence-parallel
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size ${TP}
	--pipeline-model-parallel-size ${PP}
    --context-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 1000
    --eval-interval 1000
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --log-throughput
)

python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
