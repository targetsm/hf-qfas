import argparse
import os.path as op


def parse_args_function():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--trainsplit',
        type=str,
        default='train',
        choices=['minitrain', 'train', 'all'],
        help='Amount of data to use for training, minitrain: 5k samples, train: 90k samples, all: 180k samples'
    )
    parser.add_argument(
        '--load_ckpt',
        type=str,
        default='',
        help='Load pre-trained weights from your training procedure, e.g., logs/EXP_KEY/latest.pt'
    )
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=200,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--eval_every_epoch',
        type=int,
        default=1,
        help='Evaluate your model in the training process every K training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size used by data loader'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0
    )
    parser.add_argument(
        '--drop_last',
        type=lambda x: (str(x).lower() in ['t', 'true', '1', 'y', 'yes']),
        default=True
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default='vK8cYmOLjTtfPlKA10jk8hHll',
        help='comet ml api key'
    )
    parser.add_argument(
        '--proj_name',
        type=str,
        default='nlp',
        help='comet ml project name'
    )
    args = parser.parse_args()

    root_dir = op.join('.')
    data_dir = op.join(root_dir, 'open_ai_data')
    args.root_dir = root_dir
    args.data_dir = data_dir
    args.experiment = None
    return args

args = parse_args_function()
