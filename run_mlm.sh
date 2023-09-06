export TRAIN_FILE=data/train.txt
export VALIDATION_FILE=data/dev.txt
export TRAIN_REF_FILE=data/ref_train.txt
export VALIDATION_REF_FILE=data/ref_dev.txt
export BERT_RESOURCE=/disc1/yu/puncs/chinese-roberta-wwm-ext
export OUTPUT_DIR=/disc1/yu/SCL_final
python run_mlm_wwm.py \
    --model_name_or_path $BERT_RESOURCE \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --train_ref_file $TRAIN_REF_FILE \
    --validation_ref_file $VALIDATION_REF_FILE \
    --per_device_train_batch_size 116 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 5 \
    --pad_to_max_length \
    --max_seq_length 512 \
    --evaluation_strategy steps \
    --save_steps 2000 \
    --eval_steps 2000 \
    --cache_dir cache_dir \
    --do_train \
    --do_eval \
    --save_total_limit 10 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir