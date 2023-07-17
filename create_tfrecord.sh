PYTHONPATH="research" python tools/datasets/create_coco_tf_record.py \
        --logtostderr \
        --include_masks \
        --image_dir="preprocess/test" \
        --object_annotations_file="data_original/test_coco.json" \
        --output_file_prefix="data_processed/tfrecord/test" \
        --num_shards=50