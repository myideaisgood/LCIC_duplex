import argparse

def parse_training_args(parser):
    """Add args used for training only.

    Args:
        parser: An argparse object.
    """
    parser.add_argument('--load', type=str2bool, default='yes',
                        help='Whether to load model or not')

    parser.add_argument('--gpu_num', type=int, default=0,
                        help='GPU number to use')

    # Data parameters
    parser.add_argument('--crop_size', type=int, default=512,
                        help='Size to crop the data (Must be exponential of 2')

    # Directory parameters
    parser.add_argument('--ckpt_dir', type=str, default='ckpt/',
                        help='The location of model checkpoint')

    parser.add_argument('--data_dir', type=str, default='data/',
                        help='The location of data directory')

    parser.add_argument('--tensorboard_dir', type=str, default='board/',
                        help='The location of tensorboard directory')

    # Model parameters
    parser.add_argument('--layer_num', type=int, default=4,
                        help='Layer of model')

    parser.add_argument('--hidden_unit', type=int, default=64,
                        help='Hidden units of model')

    parser.add_argument('--ctx_up', type=int, default=2,
                        help='Number of pixels up of reference pixel in context')

    parser.add_argument('--ctx_left', type=int, default=2,
                        help='Number of pixels left of reference pixel in context')

    # Session parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    parser.add_argument('--channel_epoch', type=int, default=500,
                        help='Epochs to train individual channel')

    parser.add_argument('--joint_epoch', type=int, default=500,
                        help='Epochs to train yuv channel together')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Size of batch')

    parser.add_argument('--lambda_ctx', type=float, default=1.0,
                        help='Balancing parameter between pred/context')

    parser.add_argument('--lambda_y', type=float, default=3.0,
                        help='Balancing parameter between y/u/v')                        

    parser.add_argument('--lambda_u', type=float, default=1.0,
                        help='Balancing parameter between y/u/v')                        

    parser.add_argument('--lambda_v', type=float, default=1.0,
                        help='Balancing parameter between y/u/v')                                                

    parser.add_argument('--save_every', type=float, default=100,
                        help='Interval of saving the model')

    parser.add_argument('--print_every', type=float, default=100,
                        help='Print every')


def parse_args():
    """Initializes a parser and reads the command line parameters.

    Raises:
        ValueError: If the parameters are incorrect.

    Returns:
        An object containing all the parameters.
    """

    parser = argparse.ArgumentParser(description='UNet')
    parse_training_args(parser)

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

if __name__ == '__main__':
    """Testing that the arguments in fact do get parsed
    """

    args = parse_args()
    args = args.__dict__
    print("Arguments:")

    for key, value in sorted(args.items()):
        print('\t%15s:\t%s' % (key, value))
