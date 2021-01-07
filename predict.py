import os
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pyspark.sql import SparkSession

from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row

from lib.networks import Net
from lib.schemas import MnistSchema
from lib.modules.layers import confusion_layer
from lib.modules.activation import z_score_hardmax, z_score_hardsquaremax, z_score_softmax

DEFAULT_MNIST_DATA_PATH = '/Users/PC-1/Downloads/AI-Immune-System/tmp'

os.environ["PYSPARK_SUBMIT_ARGS"] = "--master local[2] pyspark-shell"
os.environ["HADOOP_HOME"] = "C:\winutils"


def _arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--img_size", type=int, default=32,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output-url', type=str,
                        default='file://{}'.format(DEFAULT_MNIST_DATA_PATH), metavar='S',
                        help='hdfs://... or file:/// url where the parquet dataset will be written to.')
    parser.add_argument('--no_spark', action='store_true', default=True,
                        help='disables Spark and Petastorm')
    return parser


def _transform_dataset(dataset, labels):
    return [{
        'data': np.array(list(data.getdata()), dtype=np.uint8).reshape(28, 28),
        'target': target,
        'new_target': new_target
    } for (data, target), new_target in list(zip(list(dataset), labels))]


def mnist_data_to_petastorm_dataset(output_url, dataset_name, data, spark_master=None,
                                    parquet_files_count=1):
    session_builder = SparkSession \
        .builder \
        .appName('MNIST Dataset')
    if spark_master:
        session_builder.master(spark_master)

    spark = session_builder.getOrCreate()

    dset_output_url = f'{output_url}/{dataset_name}'
    print(dset_output_url)
    with materialize_dataset(spark, dset_output_url, MnistSchema, row_group_size_mb=1):
        idx_image_digit_list = map(lambda idx_image_digit: {
            MnistSchema.idx.name: idx_image_digit[0],
            MnistSchema.digit.name: idx_image_digit[1]['new_target'],
            MnistSchema.image.name: idx_image_digit[1]['data']
        }, enumerate(data))

        sql_rows = map(lambda r: dict_to_spark_row(
            MnistSchema, r), idx_image_digit_list)

        spark.createDataFrame(sql_rows, MnistSchema.as_spark_schema()) \
            .coalesce(parquet_files_count) \
            .write \
            .option('compression', 'none') \
            .parquet(dset_output_url)


def main():
    args = _arg_parser().parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if use_cuda else 'cpu')
    ngpu = torch.cuda.device_count()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    testset = datasets.MNIST("./assets/data/mnist", train=True, download=True)

    transformed_dataset = list(
        map(lambda data: (transform(data[0]), data[1]), list(testset)[0:3000]))
    test_loader = DataLoader(
        transformed_dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        model = Net().to(device)

        if use_cuda and ngpu > 1:
            model = nn.DataParallel(model, list(range(ngpu)))
        else:
            model = nn.DataParallel(model)

        model.load_state_dict(torch.load('./classifier.pt', map_location=device))
        model.eval()

        classes = list(testset.class_to_idx.values())

        labels_list = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, x = model(data)

            output = z_score_hardsquaremax(x)

            _, labels = confusion_layer(
                output, classes, len(classes))

            print(f'\nResult: {labels}\n Target: {target}')
            labels_list += labels

        if not args.no_spark:
            mnist_data_to_petastorm_dataset(
                data=_transform_dataset(list(testset), labels_list),
                output_url=args.output_url,
                dataset_name='z_score_hardmax'
            )


if __name__ == '__main__':
    main()
