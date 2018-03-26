#!/usr/bin/env bash

function usage() {
    echo "Usage: bash $0 action"
    echo "action:"
    echo "      train | test"
    exit 1
}


function args() {
    if [ $# -lt 1 ];then
        usage
    fi

    action=$1 && shift
    echo "action: ${action}"
}

function train() {
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
	--train_file=/data1/xuejiao/data/match/corpus.txt
}


function test() {
python main.py \
	--train=False \
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
	--test_file=/data1/xuejiao/data/ops/header_ops_592_corpus.txt
}

args $@

case ${action} in
    "train")
        train ;;
    "test")
        test ;;
     *)
     usage;;
esac
