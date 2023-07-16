# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Data loader and processing.

Defines input_fn of Mask-RCNN for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

"""
import tensorflow.compat.v1 as tf

from . import anchors
from . import coco_utils
from . import preprocess_ops
from . import spatial_transform_ops
from .object_detection import tf_example_decoder


MAX_NUM_INSTANCES = 100
MAX_NUM_VERTICES_PER_INSTANCE = 1500
MAX_NUM_POLYGON_LIST_LEN = 2 * MAX_NUM_VERTICES_PER_INSTANCE * MAX_NUM_INSTANCES
POLYGON_PAD_VALUE = coco_utils.POLYGON_PAD_VALUE
EVAL_IMAGE_SIZE = 512


def _prepare_labels_for_eval(data,
                             training_image_scale,
                             target_num_instances=MAX_NUM_INSTANCES,
                             num_attributes=None,
                             use_instance_mask=False,
                             gt_mask_size=112):
  """Create labels dict for infeed from data of tf.Example."""

  image = data['image']
  image_height = tf.cast(tf.shape(image)[0], tf.float32)
  image_width = tf.cast(tf.shape(image)[1], tf.float32)
  scale = tf.maximum(image_height, image_width) / tf.cast(EVAL_IMAGE_SIZE, tf.float32)
  eval_height = tf.cast(image_height / scale, tf.int32)
  eval_width = tf.cast(image_width / scale, tf.int32)
  eval_scale = training_image_scale / scale

  boxes = data['groundtruth_boxes']
  classes = data['groundtruth_classes']
  classes = tf.cast(classes, dtype=tf.float32)
  num_labels = tf.shape(classes)[0]
  boxes = preprocess_ops.pad_to_fixed_size(boxes, -1, [target_num_instances, 4])
  classes = preprocess_ops.pad_to_fixed_size(classes, -1, [target_num_instances, 1])
  is_crowd = data['groundtruth_is_crowd']
  is_crowd = tf.cast(is_crowd, dtype=tf.float32)
  is_crowd = preprocess_ops.pad_to_fixed_size(is_crowd, 0, [target_num_instances, 1])

  labels = {}
  labels['eval_width'] = eval_width
  labels['eval_height'] = eval_height
  labels['eval_scale'] = eval_scale
  labels['groundtruth_boxes'] = boxes
  labels['groundtruth_classes'] = classes
  labels['num_groundtruth_labels'] = num_labels
  labels['groundtruth_is_crowd'] = is_crowd

  if use_instance_mask:
    cropped_gt_masks = tf.image.crop_and_resize(
        image=tf.expand_dims(data['groundtruth_instance_masks'], axis=-1),
        boxes=data['groundtruth_boxes'],
        box_indices=tf.range(tf.shape(data['groundtruth_boxes'])[0], dtype=tf.int32),
        crop_size=[gt_mask_size, gt_mask_size],
        method='bilinear',
    )
    cropped_gt_masks = tf.squeeze(cropped_gt_masks, axis=-1)

    # Pads cropped_gt_masks.
    cropped_gt_masks = tf.reshape(cropped_gt_masks, tf.stack([tf.shape(cropped_gt_masks)[0], -1]))
    cropped_gt_masks = preprocess_ops.pad_to_fixed_size(cropped_gt_masks, -1, [target_num_instances, gt_mask_size ** 2])
    cropped_gt_masks = tf.reshape(cropped_gt_masks, [target_num_instances, gt_mask_size, gt_mask_size])

    # reduce size of masks by converting them to boolean type
    cropped_gt_masks = tf.greater(cropped_gt_masks, .5)

    labels['groundtruth_cropped_masks'] = cropped_gt_masks

    if 'groundtruth_area' in data:
      groundtruth_area = data['groundtruth_area']
      groundtruth_area = preprocess_ops.pad_to_fixed_size(
          groundtruth_area, 0, [target_num_instances, 1])
      labels['groundtruth_area'] = groundtruth_area

  if num_attributes:
    labels['groundtruth_attributes'] = preprocess_ops.pad_to_fixed_size(
        data['groundtruth_attributes'], -1, [target_num_instances, num_attributes])

  return labels


class InputReader(object):
  """Input reader for dataset."""

  def __init__(self,
               file_pattern,
               mode=tf.estimator.ModeKeys.TRAIN,
               num_examples=0,
               use_fake_data=False,
               use_instance_mask=False,
               max_num_instances=MAX_NUM_INSTANCES,
               max_num_polygon_list_len=MAX_NUM_POLYGON_LIST_LEN,
               num_attributes=None):
    self._file_pattern = file_pattern
    self._max_num_instances = max_num_instances
    self._max_num_polygon_list_len = max_num_polygon_list_len
    self._mode = mode
    self._num_examples = num_examples
    self._use_fake_data = use_fake_data
    self._use_instance_mask = use_instance_mask
    self._num_attributes = num_attributes

  def _create_dataset_fn(self):
    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset

    return _prefetch_dataset

  def _create_example_decoder(self):
    return tf_example_decoder.TfExampleDecoder(
        use_instance_mask=self._use_instance_mask,
        num_attributes=self._num_attributes,
    )

  def _create_dataset_parser_fn(self, params):
    """Create parser for parsing input data (dictionary)."""
    example_decoder = self._create_example_decoder()

    def _dataset_parser(value):
      """Parse data to a fixed dimension input image and learning targets.

      Args:
        value: A dictionary contains an image and groundtruth annotations.

      Returns:
        features: a dictionary that contains the image and auxiliary
          information. The following describes {key: value} pairs in the
          dictionary.
          image: Image tensor that is preproessed to have normalized value and
            fixed dimension [image_size, image_size, 3]
          image_info: image information that includes the original height and
            width, the scale of the proccessed image to the original image, and
            the scaled height and width.
          source_ids: Source image id. Default value -1 if the source id is
            empty in the groundtruth annotation.
        labels: a dictionary that contains auxiliary information plus (optional)
          labels. The following describes {key: value} pairs in the dictionary.
          `labels` is only for training.
          score_targets_dict: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors]. The height_l and width_l
            represent the dimension of objectiveness score at l-th level.
          box_targets_dict: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors * 4]. The height_l and
            width_l represent the dimension of bounding box regression output at
            l-th level.
          gt_boxes: Groundtruth bounding box annotations. The box is represented
             in [y1, x1, y2, x2] format. The tennsor is padded with -1 to the
             fixed dimension [self._max_num_instances, 4].
          gt_classes: Groundtruth classes annotations. The tennsor is padded
            with -1 to the fixed dimension [self._max_num_instances].
          cropped_gt_masks: groundtrugh masks cropped by the bounding box and
            resized to a fixed size determined by params['gt_mask_size']
      """
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        data['groundtruth_is_crowd'] = tf.cond(
            tf.greater(tf.size(data['groundtruth_is_crowd']), 0),
            lambda: data['groundtruth_is_crowd'],
            lambda: tf.zeros_like(data['groundtruth_classes'], dtype=tf.bool))
        image = data['image']
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        orig_image = image
        source_id = data['source_id']
        source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
                             source_id)
        source_id = tf.string_to_number(source_id)

        if (self._mode == tf.estimator.ModeKeys.PREDICT or
            self._mode == tf.estimator.ModeKeys.EVAL):
          image = preprocess_ops.normalize_image(image)
          if params['resize_method'] == 'retinanet':
            image, image_info, _, _, _, _ = preprocess_ops.resize_crop_pad(
                image, params['image_size'], 2 ** params['max_level'])
          else:
            image, image_info, _, _, _ = preprocess_ops.resize_crop_pad_v2(
                image, params['short_side'], params['long_side'],
                2 ** params['max_level'])
          if params['precision'] == 'bfloat16':
            image = tf.cast(image, dtype=tf.bfloat16)

          features = {
              'images': image,
              'image_info': image_info,
              'source_ids': source_id,
          }
          if params['visualize_images_summary']:
            resized_image = tf.image.resize_images(orig_image,
                                                   params['image_size'])
            features['orig_images'] = resized_image
          if (params['include_groundtruth_in_features'] or
              self._mode == tf.estimator.ModeKeys.EVAL):
            labels = _prepare_labels_for_eval(
                data,
                image_info[2],
                target_num_instances=self._max_num_instances,
                num_attributes=self._num_attributes,
                use_instance_mask=params['include_mask'],
                gt_mask_size=params['gt_mask_size'])
            return {'features': features, 'labels': labels}
          else:
            return {'features': features}

        elif self._mode == tf.estimator.ModeKeys.TRAIN:
          instance_masks = None
          if self._use_instance_mask:
            instance_masks = data['groundtruth_instance_masks']

          attributes = None
          if self._num_attributes:
              attributes = data['groundtruth_attributes']

          boxes = data['groundtruth_boxes']
          classes = data['groundtruth_classes']
          classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
          if not params['use_category']:
            classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)

          if (params['skip_crowd_during_training'] and
              self._mode == tf.estimator.ModeKeys.TRAIN):
            indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
            classes = tf.gather_nd(classes, indices)
            boxes = tf.gather_nd(boxes, indices)

            if self._use_instance_mask:
              instance_masks = tf.gather_nd(instance_masks, indices)

            if self._num_attributes:
              attributes = tf.gather_nd(attributes, indices)

          image = preprocess_ops.normalize_image(image)
          if params['input_rand_hflip']:
            flipped_results = (
                preprocess_ops.random_horizontal_flip(
                    image, boxes=boxes, masks=instance_masks))
            if self._use_instance_mask:
              image, boxes, instance_masks = flipped_results
            else:
              image, boxes = flipped_results
          # Scaling, jittering and padding.
          if params['resize_method'] == 'retinanet':
            image, image_info, boxes, classes, attributes, cropped_gt_masks = (
                preprocess_ops.resize_crop_pad(
                    image,
                    params['image_size'],
                    2 ** params['max_level'],
                    aug_scale_min=params['aug_scale_min'],
                    aug_scale_max=params['aug_scale_max'],
                    boxes=boxes,
                    classes=classes,
                    attributes=attributes,
                    masks=instance_masks,
                    crop_mask_size=params['gt_mask_size']))
          else:
            image, image_info, boxes, classes, cropped_gt_masks = (
                preprocess_ops.resize_crop_pad_v2(
                    image,
                    params['short_side'],
                    params['long_side'],
                    2 ** params['max_level'],
                    aug_scale_min=params['aug_scale_min'],
                    aug_scale_max=params['aug_scale_max'],
                    boxes=boxes,
                    classes=classes,
                    masks=instance_masks,
                    crop_mask_size=params['gt_mask_size']))
          if cropped_gt_masks is not None:
            cropped_gt_masks = tf.pad(
                cropped_gt_masks,
                paddings=tf.constant([[0, 0,], [2, 2,], [2, 2]]),
                mode='CONSTANT',
                constant_values=0.)

          padded_height, padded_width, _ = image.get_shape().as_list()
          padded_image_size = (padded_height, padded_width)
          input_anchors = anchors.Anchors(
              params['min_level'],
              params['max_level'],
              params['num_scales'],
              params['aspect_ratios'],
              params['anchor_scale'],
              padded_image_size)
          anchor_labeler = anchors.AnchorLabeler(
              input_anchors,
              params['num_classes'],
              params['rpn_positive_overlap'],
              params['rpn_negative_overlap'],
              params['rpn_batch_size_per_im'],
              params['rpn_fg_fraction'])

          # Assign anchors.
          score_targets, box_targets = anchor_labeler.label_anchors(
              boxes, classes)

          # Pad groundtruth data.
          boxes = preprocess_ops.pad_to_fixed_size(
              boxes, -1, [self._max_num_instances, 4])
          classes = preprocess_ops.pad_to_fixed_size(
              classes, -1, [self._max_num_instances, 1])

          # Pads cropped_gt_masks.
          if self._use_instance_mask:
            cropped_gt_masks = tf.reshape(
                cropped_gt_masks, tf.stack([tf.shape(cropped_gt_masks)[0], -1]))
            cropped_gt_masks = preprocess_ops.pad_to_fixed_size(
                cropped_gt_masks, -1,
                [self._max_num_instances, (params['gt_mask_size'] + 4) ** 2])
            cropped_gt_masks = tf.reshape(
                cropped_gt_masks,
                [self._max_num_instances, params['gt_mask_size'] + 4,
                 params['gt_mask_size'] + 4])

          # pad attributes
          if self._num_attributes:
            attributes = preprocess_ops.pad_to_fixed_size(
                attributes, -1, [self._max_num_instances, self._num_attributes])

          if params['precision'] == 'bfloat16':
            image = tf.cast(image, dtype=tf.bfloat16)

          features = {
              'images': image,
              'image_info': image_info,
              'source_ids': source_id,
          }
          labels = {}
          for level in range(params['min_level'], params['max_level'] + 1):
            labels['score_targets_%d' % level] = score_targets[level]
            labels['box_targets_%d' % level] = box_targets[level]
          labels['gt_boxes'] = boxes
          labels['gt_classes'] = classes

          if self._use_instance_mask:
            labels['cropped_gt_masks'] = cropped_gt_masks

          if self._num_attributes:
              labels['gt_attributes'] = attributes

          return features, labels

    return _dataset_parser

  def __call__(self, params, input_context=None):
    dataset_parser_fn = self._create_dataset_parser_fn(params)
    dataset_fn = self._create_dataset_fn()
    batch_size = params['batch_size'] if 'batch_size' in params else 1
    dataset = tf.data.Dataset.list_files(
        self._file_pattern, shuffle=(self._mode == tf.estimator.ModeKeys.TRAIN))
    if input_context is not None:
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.repeat()

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            dataset_fn,
            cycle_length=32,
            sloppy=(self._mode == tf.estimator.ModeKeys.TRAIN)))
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(64)

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            dataset_parser_fn,
            batch_size=batch_size,
            num_parallel_batches=64,
            drop_remainder=True))

    # Enable TPU performance optimization: transpose input, space-to-depth
    # image transform, or both.
    if (self._mode == tf.estimator.ModeKeys.TRAIN and
        (params['transpose_input'] or
         (params['backbone'].startswith('resnet') and
          params['conv0_space_to_depth_block_size'] > 0))):

      def _transform_images(features, labels):
        """Transforms images."""
        images = features['images']
        if (params['backbone'].startswith('resnet') and
            params['conv0_space_to_depth_block_size'] > 0):
          # Transforms images for TPU performance.
          features['images'] = (
              spatial_transform_ops.fused_transpose_and_space_to_depth(
                  images,
                  params['conv0_space_to_depth_block_size'],
                  params['transpose_input']))
        else:
          features['images'] = tf.transpose(features['images'], [1, 2, 3, 0])
        return features, labels

      dataset = dataset.map(_transform_images, num_parallel_calls=16)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if self._num_examples > 0:
      dataset = dataset.take(self._num_examples)
    if self._use_fake_data:
      # Turn this dataset into a semi-fake dataset which always loop at the
      # first batch. This reduces variance in performance and is useful in
      # testing.
      dataset = dataset.take(1).cache().repeat()
    return dataset