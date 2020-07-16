"""
Convert a label map file in .pbtxt format to text format.

The output should be used as the label file for the edge tpu model. 

Copyright (c) 2019 Lindo St. Angel
"""

import argparse
from object_detection.utils import label_map_util

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str,
    default='/home/lindo/develop/tensorflow/models/annotations/label_map.pbtxt',
    help='path to input label map file in .pbtxt format')
ap.add_argument('-o', '--output', type=str,
    default='/home/lindo/develop/tensorflow/models/tflite_models/label_map.txt',
    help='path to output label map file in .txt format')
args = vars(ap.parse_args())

# Convert from .pbtxt to .txt.
label_map_dict = label_map_util.get_label_map_dict(args['input'])

# Save file.
with open(args['output'], 'w') as f:
    for i, j in label_map_dict.items():
        f.write(str(j - 1) + ' ')
        f.write(str(i) + '\n')

print('Wrote label map text file to {}.'.format(args['output']))