#!/bin/bash


export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
# Change for multinode config
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=14535 # 適当な値
export LOCAL_RANK=$((OMPI_COMM_WORLD_RANK % GPUS_PER_NODE))

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2048

# 継続事前学習を行う場合は以下の並列数を変更すると再度新しい並列数でモデル変換が必要となります
TP=1  # --tensor-model-parallel-size
PP=1  # --pipeline-model-parallel-size

CHECKPOINT_LOAD_PATH=$1
CHECKPOINT_SAVE_PATH=$1
TOKENIZER_SAVE_PATH=$2/tokenizer.model
TENSORBOARD_LOGS_PATH=$1/tensorboard
DATA_PATH=llama3_arxiv_text_document

GPT_MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --seq-length 4096
    --no-masked-softmax-fusion
    --max-position-embeddings 8192
    --attention-dropout 0
    --hidden-dropout 0
    --normalization RMSNorm
    --ffn-hidden-size 14336
    --num-query-groups 8
    --swiglu
    --group-query-attention
    --tokenizer-type Llama3Tokenizer
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --disable-bias-linear
    --tokenizer-model $TOKENIZER_SAVE_PATH
    --rotary-base 500000
    --rotary-percent 1.0
    --attention-softmax-in-fp32
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 500000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --fp16
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 430000
    --use-flash-attn
    --fp8-format 'hybrid'
    --recompute-granularity "selective"
    --transformer-impl "transformer_engine"
    --use-distributed-optimizer
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size ${TP}
	--pipeline-model-parallel-size ${PP}
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --save $CHECKPOINT_SAVE_PATH
    --load $CHECKPOINT_LOAD_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --log-throughput
    --exit-on-missing-checkpoint
    --use-checkpoint-args
    --no-load-optim
    --no-load-rng
)

python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
