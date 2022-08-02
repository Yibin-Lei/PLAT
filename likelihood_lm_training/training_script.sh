#!/bin/bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file yelp_train.txt \
    --validation_file yelp_valid.txt \
    --do_train \
    --do_eval \
    --output_dir ./yelp_LM  \
    --line_by_line \
    --per_device_train_batch_size 8 \
    --num_train_epochs  5 \
    --save_steps 10000 \
    --fp16  \
    --tokenizer_name ./yelp_tokenizer \
    --use_fast_tokenizer False

