"""
Convert dataset to TFRecord for TF object detection training.

Example usage:
    python3 create_tf_record.py \
        --root_dir ./ \
        --image_dir images \
        --annotation_dir annotations \
        --output_dir tf-record \
        --dataset_name radar-ml

    Only datset_name is required.

Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

Copyright (c) 2019~2020 Lindo St. Angel
"""

import hashlib
import io
import logging
import os
import random
import re
import contextlib2
import numpy as np
import PIL.Image
import tensorflow as tf
import argparse
from lxml import etree
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

logger = logging.getLogger(__name__)

NUM_TFRECORD_SHARDS = 1
TRAIN_VAL_SPLIT = 0.8
TFRECORD_TRAIN_NAME = 'train'
TFRECORD_VAL_NAME = 'val'

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
        data: dict holding PASCAL XML fields for a single image (obtained by
            running dataset_util.recursive_parse_xml_to_dict)
        label_map_dict: A map from string label names to integers ids.
        image_subdirectory: String specifying subdirectory within the
            Pascal dataset directory holding the actual image data.
        ignore_difficult_instances: Whether to skip difficult instances in the
            dataset  (default: False).

    Returns:
        example: The converted tf.Example.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = os.path.join(image_subdirectory, data['filename'])

    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)

        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format must be JPEG.')

        key = hashlib.sha256(encoded_jpg).hexdigest()

        width = int(data['size']['width'])
        height = int(data['size']['height'])

        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []

        if 'object' in data:
            for obj in data['object']:
                difficult = bool(int(obj['difficult']))
                if ignore_difficult_instances and difficult:
                    continue

            difficult_obj.append(int(difficult))

            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])

            xmins.append(xmin / width)
            ymins.append(ymin / height)
            xmaxs.append(xmax / width)
            ymaxs.append(ymax / height)
            #class_name = get_class_name_from_filename(data['filename'])
            class_name = obj['name']
            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[class_name])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

        feature_dict = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
    """Creates a TFRecord file from examples.

    Args:
        output_filename: Path to where output file is saved.
        num_shards: Number of shards for output file.
        label_map_dict: The label map dictionary.
        annotations_dir: Directory where annotation files are stored.
        image_dir: Directory where image files are stored.
        examples: Examples to parse and save to tf record.
    """

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filename, num_shards)

        for idx, example in enumerate(examples):
            if idx % 10 == 0:
                logger.info('On image %d of %d', idx, len(examples))

            xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')

            if not os.path.exists(xml_path):
                logger.warning('Could not find %s, ignoring example.', xml_path)
                continue

            with tf.io.gfile.GFile(xml_path, 'r') as fid:
                xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                try:
                    tf_example = dict_to_tf_example(
                        data,
                        label_map_dict,
                        image_dir)

                    if tf_example:
                        shard_idx = idx % num_shards

                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
                except ValueError:
                    logger.warning('Invalid example: %s, ignoring.', xml_path)

def gen_trainval_list(images_path):
    """Creates a list of image names without file extensions.
        The list items will not match the ordering of the images on disk.

    Args:
        images_path: Path to where images are located.
    """
    def make(file):
        if file.endswith('.jpg' or '.jpeg'):
            return os.path.basename(file).split('.')[0]
    return [make(file) for file in os.listdir(images_path)]

def main(args):
    logger.info('Reading dataset info.')
    image_dir = os.path.join(args.root_dir, args.image_dir,
        args.dataset_name)
    logger.info(f'Image directory: {image_dir}')
    annotations_dir = os.path.join(args.root_dir, args.annotation_dir,
        args.dataset_name)
    logger.info(f'Annotation directory: {annotations_dir}')
    label_map = os.path.join(args.root_dir, args.annotation_dir,
        args.dataset_name, args.label_map_name)
    logger.info(f'Label map: {label_map}')

    # Split data into training and validation sets.
    random.seed(42)
    examples_list = gen_trainval_list(image_dir)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(TRAIN_VAL_SPLIT * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    logger.info('Found %d training and %d validation examples.',
        len(train_examples), len(val_examples))

    train_output_path = os.path.join(args.root_dir, args.output_dir,
        args.dataset_name, TFRECORD_TRAIN_NAME)
    val_output_path = os.path.join(args.root_dir, args.output_dir,
        args.dataset_name, TFRECORD_VAL_NAME)

    label_map_dict = label_map_util.get_label_map_dict(label_map)
  
    # Create training TFRecord.
    logger.info('Creating training TFRecord.')
    create_tf_record(
        train_output_path,
        NUM_TFRECORD_SHARDS,
        label_map_dict,
        annotations_dir,
        image_dir,
        train_examples)
    logger.info(f'Created training TFRecord: {train_output_path}')

    # Create validation TFRecord.
    logger.info('Creating validation TFRecord.')
    create_tf_record(
        val_output_path,
        NUM_TFRECORD_SHARDS,
        label_map_dict,
        annotations_dir,
        image_dir,
        val_examples)
    logger.info(f'Created validation TFRecord: {val_output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
        help='Root directory.',
        default='./')
    parser.add_argument('--output_dir', type=str,
        help='TFRecord directory.',
        default='tf-record')
    parser.add_argument('--annotation_dir', type=str,
        help='Annotation directory.',
        default='annotations')
    parser.add_argument('--label_map_name', type=str,
        help='Label map name.',
        default='label_map.pbtxt')
    parser.add_argument('--image_dir', type=str,
        help='Image directory.',
        default='images')
    parser.add_argument('--dataset_name', type=str,
        help='Name of dataset',
        required=True)
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        level=logging.DEBUG)

    main(args)