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
	--config_file=/data1/xuejiao/model/ner/config_file \
	--pre_emb=False \
	--tag_schema=iob \
	--max_epoch=50 \
	--max_sentence=10000 \
	--model_type=idcnn \
	--emb_file=/data1/xuejiao/data/embedding/vec.txt \
	--tag_file=/data1/xuejiao/data/entity_tag.txt \
	--train_folder=/data1/xuejiao/data/wiki/corpus/ \
	--train_file=/data1/xuejiao/data/match/corpus.txt \
	--test_file=/data1/xuejiao/data/idcnn-crf-data/example.test
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
	--config_file=/data1/xuejiao/model/ner/config_file \
	--pre_emb=False \
	--tag_schema=iob \
	--max_epoch=50 \
	--max_sentence=50000 \
	--model_type=idcnn \
	--emb_file=/data1/xuejiao/data/embedding/vec.txt \
	--tag_file=/data1/xuejiao/data/entity_tag.txt \
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
