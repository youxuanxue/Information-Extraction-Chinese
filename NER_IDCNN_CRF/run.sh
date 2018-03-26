#!/usr/bin/env bash

python main.py \
	--train=True \
	--clean=True \
	--ckpt_path=/data1/xuejiao/model/ner/ckpt \
	--summary_path=/data1/xuejiao/model/ner/summary \
	--log_file=/data1/xuejiao/model/ner/train.log \
	--map_file=/data1/xuejiao/model/ner/maps.pkl \
	--vocab_file=/data1/xuejiao/model/ner/vocab.json \
	--result_path=/data1/xuejiao/model/ner/result/  \
	--emb_file=/data1/xuejiao/model/ner/vec.txt \
	--pre_emb=False \
	--tag_schema=iob \
	--max_epoch=50 \
	--model_type=idcnn \
	--train_file=/data1/xuejiao/data/match/corpus.txt \
    --test_file=/data1/xuejiao/data/ops/header_ops_592_corpus.txt