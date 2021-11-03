import os
import sys
import json
from pathlib import Path
import pandas as pd
import random
import tensorflow as tf
import io
import argparse
from PIL import Image
from collections import namedtuple

from object_detection.utils import dataset_util, label_map_util

import logging
logging.basicConfig(stream=sys.stdout, format='',
                level=logging.INFO, datefmt=None)
logger = logging.getLogger('NJDD-prepare-data')


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow json-to-TFRecord converter")
parser.add_argument("-json",
                    "--json_path",
                    help="Path to the input .json files.",
                    type=str)
# parser.add_argument("-subset",
#                     "--subset",
#                     help="Type of the subset: train, validation, test", type=str)             
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str)
parser.add_argument("-o",
                    "--output_dir",
                    help="Path of the output dir for storing TFRecord (.record) file.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as JSON_DIR.",
                    type=str, default=None)
parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)

args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.json_dir

label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)

def bbox_dict_to_df(bbox_dict):
    """
    This function assumes that the objects list contains one element (v['objects'][0])
    """
    log_index = 'bbox_dict_to_df>'
    df_ls = []
    for k, v in bbox_dict.items():
        filename = k
        height = v['size']['height']
        width = v['size']['width']
        ym = v['objects'][0]['bbox'][0]
        xm = v['objects'][0]['bbox'][1]
        yM = v['objects'][0]['bbox'][2]
        xM = v['objects'][0]['bbox'][3]
        values = (filename, height, width, ym, xm, yM, xM, v['objects'][0]['name'])
        df_ls.append(values)
    logger.info(f'{log_index} Collected {len(df_ls)} objects')
    df = pd.DataFrame(df_ls, columns=['fname', 'height', 'width', 'ym', 'xm', 'yM', 'xM', 'class'])
    return df

def split_dataset(df, perc=0.9):
    log_index = 'split_dataset>'
    df = df.sample(frac=1).reset_index(drop=True)
    num_train = int(perc * len(df))
    df_train = df.iloc[0:num_train]
    df_val = df.iloc[num_train:]
    logger.info(f'{log_index} TRAINING EXAMPLES: {len(df_train)} - VALIDATION EXAMPLES: {len(df_val)}')
    return df_train, df_val


def class_text_to_int(row_label):
     return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['fname', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    log_index = 'create_tf_example>'
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.fname)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    logger.info(f'{log_index} Retrived image with size: {width, height} - (w,h)')

    filename = group.fname.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.object.iterrows():
        xmins.append(row['xm'])
        xmaxs.append(row['xM'])
        ymins.append(row['ym'])
        ymaxs.append(row['yM'])
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
    logger.info(f'{log_index} Collected {len(xmins)} rows')

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    log_index = 'main>'
    logger.info(f'{log_index} Reading bbox at {args.json_path} ...')
    with open(args.json_path, 'r') as f:
        bbox_dict = json.load(f)
    df = bbox_dict_to_df(bbox_dict)
    df_train, df_val = split_dataset(df)
    for df_s, subset in zip([df_train, df_val], ['train', 'val']):
        logger.info(f'{log_index} Writing TFRecords for subset: {subset}')
        writer = tf.io.TFRecordWriter(os.path.join(args.output_dir, f'dywidag_{subset}.records'))
        path = Path(args.image_dir)
        grouped = split(df_s, 'fname')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        logger.info(f'{log_index} Successfully created the TFRecord file: {args.output_dir}')
        if args.csv_path is not None:
            df.to_csv(args.csv_path, index=None)
            logger.info(f'{log_index} Successfully created the CSV file: {args.csv_path}')

if __name__ == '__main__':
    main()