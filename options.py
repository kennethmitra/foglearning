import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_name',
        type=str,
        default='Noniid_1_sharer',
        help='string for the name of the run'
    )
    # dataset and model
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='name of the dataset: mnist, cifar10'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='cnn_complex',  #'lenet',  # 'logistic',
        help='name of model. mnist: logistic, lenet; cifar10: cnn_tutorial, cnn_complex'
    )
    parser.add_argument(
        '--input_channels',
        type=int,
        default=3,  # <---- IMPORTANT!!!
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
    parser.add_argument(
        '--iid',
        type=bool,
        default=True,
        help="shuffle the order of the dataset for each client"
    )
    # nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='batch size when trained on client'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=32,
        help='number of local gradient update steps (tau_1)'
    )
    parser.add_argument(
        '--num_share_rounds',
        type=int,
        default=1,
        help='number of weight sharing steps (tau_2)'
    )
    parser.add_argument(
        '--num_share_devices',
        type=int,
        default=1,
        help='Number of devices to share with at each share round'
    )
    parser.add_argument(
        '--fed_avg',
        type=bool,
        default=False,
        help='Perform federate averaging. If true, no decentralized training will be don'
    )
    parser.add_argument(
        '--num_total_rounds',
        type=int,
        default=150,
        help='Number of total (train + share) rounds to perform'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=1, #'1'
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
    parser.add_argument(
        '--num_devices',
        type=int,
        default=10,
        help='number of all available devices'
    )
    parser.add_argument(
        '--comm_reliability',
        type=float,
        default=1.0,
        help='Fraction of time communication works'
    )
    parser.add_argument(
        '--model_share_strategy',
        type=str,
        default='random',
        help='random or distance',
        choices=['random', 'distance']
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (defaul: 1)'
    )
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=True,
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
