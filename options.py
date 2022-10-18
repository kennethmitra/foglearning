import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        help='name of the dataset: mnist, cifar10'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='lenet',
        help='name of model. mnist: logistic, lenet; cifar10: cnn_tutorial, cnn_complex'
    )
    parser.add_argument(
        '--input_channels',
        type=int,
        default=1,
        help='input channels. mnist:1, cifar10 :3'
    )
    parser.add_argument(
        '--output_channels',
        type=int,
        default=10,
        help='output channels'
    )
    parser.add_argument(
        '--shuffle_dataset',
        type=bool,
        default=True,
        help="shuffle the order of the dataset for each client"
    )
    # nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='batch size when trained on client'
    )
    parser.add_argument(
        '--num_communication',
        type=int,
        default=1,
        help='number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=32,
        help='number of local update (tau_1)'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type=int,
        default=1,
        help='number of edge aggregation (tau_2)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        default='1',
        help='lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type=int,
        default=1,
        help='lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0,
        help='SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='verbose for print progress bar'
    )
    # setting for federeated learning
    parser.add_argument(
        '--frac',
        type=float,
        default=1,
        help='fraction of participated clients'
    )
    parser.add_argument(
        '--num_devices',
        type=int,
        default=10,
        help='number of all available devices'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (defaul: 1)'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='data',
        help='dataset root folder'
    )
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help='Use gpu or not'
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )

    args = parser.parse_args()
    return args
