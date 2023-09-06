export LTP_RESOURCE=/disc1/models/ltp-base
export BERT_RESOURCE=/disc1/models/chinese-roberta-wwm-ext
export TRAIN_FILE=data/train.txt
export VALIDATION_FILE=data/dev.txt
export TRAIN_REF_FILE=data/ref_train.txt
export VALIDATION_REF_FILE=data/ref_dev.txt
export OUTPUT_DIR=scl_output
export CUDA_VISIBLE_DEVICES=3

python run_chinese_ref.py \
    --file_name=$TRAIN_FILE \
    --ltp=$LTP_RESOURCE \
    --bert=$BERT_RESOURCE \
    --save_path=$TRAIN_REF_FILE

python run_chinese_ref.py \
    --file_name=$VALIDATION_FILE \
    --ltp=$LTP_RESOURCE \
    --bert=$BERT_RESOURCE \
    --save_path=$VALIDATION_REF_FILE

python run_mlm_wwm.py \
    --model_name_or_path $BERT_RESOURCE \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --train_ref_file $TRAIN_REF_FILE \
    --validation_ref_file $VALIDATION_REF_FILE \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --cache_dir cache_dir \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir