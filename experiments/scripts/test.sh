#!/usr/bin/env bash
GPU_ID=$1
EXP_DIR=$2
FN=$3
NET=$4
INFERENCE_ITER=$5
NUM_IMG=$6
MODE=$7

echo using GPU: $GPU_ID
echo using checkpoint dir: $EXP_DIR
echo using ckpt: $FN
echo using net: $NET
echo using number of inference: $INFERENCE_ITER
echo using number of images: $NUM_IMG
echo using mode: $MODE
# log
OUTPUT=checkpoints/$EXP_DIR
TF_LOG=checkpoints/$EXP_DIR/tf_logs
# rm -rf ${OUTPUT}/logs/
# rm -rf ${TF_LOG}
mkdir -p ${OUTPUT}/test_logs
LOG="$OUTPUT/test_logs/`date +'%Y-%m-%d_%H:%M:%S'`"
export CUDA_VISIBLE_DEVICES=$GPU_ID
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu $GPU_ID --weights checkpoints/$EXP_DIR/$FN.ckpt --imdb imdb_1024.h5 --roidb VG-SGG --rpndb proposals.h5 --cfg experiments/cfgs/sparse_graph.yml --network $NET --inference_iter $INFERENCE_ITER --test_size $NUM_IMG --test_mode $MODE
echo Logged output to "$LOG"
