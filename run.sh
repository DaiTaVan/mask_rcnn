#!/bin/bash
rm data_processed/test_coco.json
cp data_original/test_coco.json data_processed

if [ -n "{{MODEL}}" ] && [ -n "{{DATASET_PREFIX}}" ]  && [ -n "{{OUTPUT_DIR}}" ] && [ -n "{{TPU}}" ] && [ -n "{{STEP}}" ]; then
    export STORAGE_BUCKET=gs://kaggle-imaterialist2020-data-europe-west4
    export MODEL_DIR="/media/tavandai/MAIN_WORKING_UBUNTU/Startup/image_tagging/datasets/imaterialist-fashion-2021-fgvc8/mask_rcnn/output"
    export FILE_PATTERN="/media/tavandai/MAIN_WORKING_UBUNTU/Startup/image_tagging/datasets/imaterialist-fashion-2021-fgvc8/mask_rcnn/data_processed/tfrecord/test-*"
    export OUTPUT_DIR=${STORAGE_BUCKET}/predictions/{{OUTPUT_DIR}}/{{MODEL}}_{{STEP}}

    PYTHONPATH=tf_tpu_models python tf_tpu_models/official/detection/main.py \
    --model="mask_rcnn" \
    --model_dir=${MODEL_DIR} \
    --mode=predict \
    --predict_checkpoint_step={{STEP}} \
    --predict_file_pattern=$FILE_PATTERN \
    --predict_output_dir=$OUTPUT_DIR \
    --config_file=${MODEL_DIR}/params.yaml \
    --use_tpu=True \
    --tpu={{TPU}}
else
    echo "MODEL, CONFIG, DATASET, TPU and STEP parameters are required."
fi