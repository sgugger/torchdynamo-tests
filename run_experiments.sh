#!/bin/bash

if [[ -z $1 ]];
then 
    echo "model_name_or_path not passed"
    exit 1
else
    echo "model_name_or_path = $1"
fi

if [[ -z $2 ]];
then 
    echo "num_runs not passed"
    exit 1
else
    echo "num_runs = $2"
fi

if [[ -z $3 ]];
then 
    echo "task_name not passed"
    exit 1
else
    echo "task_name = $2"
fi

model_name_or_path = $1
num_runs = $2
task_name = $3

for i in {0.. $num_runs }
do
    echo "experiment run $i"
    
    case $task_name in
        "text_classification")
            echo "Running text_classification"
            echo "inductor backend with fp32"
            accelerate launch scripts/text_classification.py \
            --task_name mrpc \
            --seed $i \
            --model_name_or_path $model_name_or_path \
            --dynamo_backend inductor
            echo "no backend with fp32"
            accelerate launch scripts/text_classification.py \
            --task_name mrpc \
            --seed $i \
            --model_name_or_path $model_name_or_path
            echo "inductor backend with fp16"
            accelerate launch scripts/text_classification.py \
            --task_name mrpc \
            --seed $i \
            --model_name_or_path $model_name_or_path \
            --dynamo_backend inductor \
            --mixed_precision fp16
            echo "no backend with fp16"
            accelerate launch scripts/text_classification.py \
            --task_name mrpc \
            --seed $i \
            --model_name_or_path $model_name_or_path \
            --mixed_precision fp16 \
            ;;
        "language_modeling")
            echo "Running language_modeling"
            echo "inductor backend with fp32"
            accelerate launch scripts/language_modeling.py \
            --dataset_name wikitext \
            --dataset_config_name wikitext-2-raw-v1 \
            --seed $i \
            --model_name_or_path $model_name_or_path \
            --dynamo_backend inductor
            echo "no backend with fp32"
            accelerate launch scripts/language_modeling.py \
            --dataset_name wikitext \
            --dataset_config_name wikitext-2-raw-v1 \
            --seed $i \
            --model_name_or_path $model_name_or_path
            echo "inductor backend with fp16"
            accelerate launch scripts/language_modeling.py \
            --dataset_name wikitext \
            --dataset_config_name wikitext-2-raw-v1 \
            --seed $i \
            --model_name_or_path $model_name_or_path \
            --dynamo_backend inductor \
            --mixed_precision fp16
            echo "no backend with fp16"
            accelerate launch scripts/language_modeling.py \
            --dataset_name wikitext \
            --dataset_config_name wikitext-2-raw-v1 \
            --seed $i \
            --model_name_or_path $model_name_or_path \
            --mixed_precision fp16
            ;;
            "cv_classification")
            echo "Running cv_classification"
            echo "inductor backend with fp32"
            accelerate launch scripts/cv_classification.py \
            --dataset_name beans \
            --seed $i \
            --model_name_or_path $model_name_or_path \
            --dynamo_backend inductor
            echo "no backend with fp32"
            accelerate launch scripts/cv_classification.py \
            --dataset_name beans \
            --seed $i \
            --model_name_or_path $model_name_or_path
            echo "inductor backend with fp16"
            accelerate launch scripts/cv_classification.py \
            --dataset_name beans \
            --seed $i \
            --model_name_or_path $model_name_or_path \
            --dynamo_backend inductor \
            --mixed_precision fp16
            echo "no backend with fp16"
            accelerate launch scripts/cv_classification.py \
            --dataset_name beans \
            --seed $i \
            --model_name_or_path $model_name_or_path \
            --mixed_precision fp16
            ;;
        *)
            echo "Invalid task_name"
            exit 1
            ;;
    esac
done



