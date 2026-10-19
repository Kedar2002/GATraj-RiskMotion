'''
Train script
Author: Mengmeng Liu
Date: 2022/09/24
'''
import argparse
import ast
from Processor import *
import numpy as np
import random
import torch
import yaml

def get_parser():
    parser = argparse.ArgumentParser(description='GATraj')
    parser.add_argument(
        '--ifvalid',default=True,type=ast.literal_eval,
        help="=False,use all train set to train,"
             "=True,use train set to train and valid")
    parser.add_argument(
        '--mlp_decoder',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--input_offset',default=True,type=ast.literal_eval)
    parser.add_argument(
        '--input_position',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--input_mix',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--ifGaussian',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--SR',default=True,type=ast.literal_eval)
    parser.add_argument("--pass_time",
                        default=2,
                        type=int)
    parser.add_argument("--final_mode",
                        default=20,
                        type=int)
    parser.add_argument("--T_max",
                        default=1000,
                        type=int)
    parser.add_argument("--eta_min",
                        default=3e-6,
                        type=float)
    parser.add_argument(
        '--output_size',default=2,type=int)
    parser.add_argument(
        '--input_size',default=2,type=int)
    parser.add_argument(
        '--min_obs',default=8,type=int)
    parser.add_argument('--num_pred', type=int, default=1, help='This is the number of predictions for each agent')
    parser.add_argument('--ratio', type=float, default=0.95, help='The overlap ratio of coexisting for group detection')
    parser.add_argument('--z_dim', type=int, default=32, help='This is the size of the latent variable')
    parser.add_argument('--hidden_size', type=int, default=64, help='The size of LSTM hidden state')
    parser.add_argument('--x_encoder_layers', type=int, default=3, help='Number of transformer block layers for x_encoder')
    parser.add_argument('--x_encoder_head', type=int, default=8, help='Head number of x_encoder')
    parser.add_argument(
        '--gpu', default=0,type=int,
        help='gpu id')
    parser.add_argument(
        '--using_cuda',default=True,type=ast.literal_eval) # We did not test on cpu
    # You may change these arguments (model selection and dirs)
    parser.add_argument(
        '--test_set',default=1,type=int,
        help='Set this value to 0~4 for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument(
        '--base_dir',default='.',
        help='Base directory including these scrits.')
    parser.add_argument(
        '--save_base_dir',default='./savedata/',
        help='Directory for saving caches and models.')
    parser.add_argument(
        '--phase', default='train',
        help='Set this value to \'train\' or \'test\'')
    parser.add_argument(
        '--GT',default=True)
    parser.add_argument(
        '--train_model', default=None,
        help='Optional experiment name; by default derived from modules and seed')
    parser.add_argument(
        '--load_model', default=0,type=int,
        help="load model weights from this index before training or testing")
    parser.add_argument(
        '--epochs_per_run', default=0, type=int,
        help='Optional number of epochs to run before stopping; num_epochs remains the full schedule length')
    parser.add_argument(
        '--seed', default=1, type=int,
        help='Random seed for training and deterministic data ordering')
    parser.add_argument('--motion_module', default=False, type=ast.literal_eval)
    parser.add_argument('--relative_velocity', default=True, type=ast.literal_eval)
    parser.add_argument('--closest_approach', default=True, type=ast.literal_eval)
    parser.add_argument('--confidence_gate', default=True, type=ast.literal_eval)
    parser.add_argument('--adaptive_neighbors', default=False, type=ast.literal_eval)
    parser.add_argument('--fix_attention', default=True, type=ast.literal_eval,
                        help='Shared baseline fix: learned scalar attention and explicit masks')
    parser.add_argument('--velocity_window', default=3, type=int)
    parser.add_argument('--velocity_scale', default=0.5, type=float,
                        help='Feature scale in metres per sampled frame')
    parser.add_argument('--risk_radius', default=2.0, type=float,
                        help='Closest-approach feature scale and optional neighbor threshold, metres')
    parser.add_argument('--adaptive_radius', default=15.0, type=float,
                        help='Maximum current distance for optional additional neighbors, metres')
    parser.add_argument(
        '--model', default='models.GATraj')
    ######################################
    parser.add_argument(
        '--dataset',default='eth5')
    parser.add_argument(
        '--save_dir')
    parser.add_argument(
        '--model_dir')
    parser.add_argument(
        '--config')
    parser.add_argument(
        '--val_fraction',default=0,type=float)
    #Perprocess
    parser.add_argument(
        '--seq_length',default=20,type=int)
    parser.add_argument(
        '--obs_length',default=8,type=int)
    parser.add_argument(
        '--pred_length',default=12,type=int)
    parser.add_argument(
        '--batch_size',default=64,type=int)
    parser.add_argument(
        '--show_step',default=40,type=int)
    parser.add_argument(
        '--step_ratio',default=0.5,type=int)
    parser.add_argument(
        '--lr_step',default=20,type=int)
    parser.add_argument(
        '--num_epochs',default=1000,type=int)
    parser.add_argument(
        '--ifshow_detail',default=True,type=ast.literal_eval)
    parser.add_argument(
        '--randomRotate',default=True,type=ast.literal_eval,
        help="=True:random rotation of each trajectory fragment")
    parser.add_argument(
        '--neighbor_thred',default=10,type=int)
    parser.add_argument(
        '--learning_rate',default=1e-04,type=float)
    parser.add_argument(
        '--clip',default=10,type=int)
    return parser
def load_arg(p, parser):
    # save arg
    if  os.path.exists(p.config):
        with open(p.config, 'r') as f:
            # default_arg = yaml.load(f,Loader=yaml.FullLoader)
            default_arg = yaml.safe_load(f)

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s=1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False
def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)
def prepare_seed(rand_seed):
	# Seed every RNG used by training and data preprocessing.
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed_all(rand_seed)
	if hasattr(torch.backends, 'cudnn'):
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

def experiment_name(args):
    if args.motion_module:
        variant = 'RiskMotion_v{}_r{}_c{}_a{}'.format(
            int(args.relative_velocity), int(args.closest_approach),
            int(args.confidence_gate), int(args.adaptive_neighbors))
    else:
        variant = 'GATraj_baseline'
    return '{}_attn{}_s{}'.format(variant, int(args.fix_attention), args.seed)
	
def parse_run_args(default_phase='train'):
    parser = get_parser()
    parser.set_defaults(phase=default_phase)
    p = parser.parse_args()
    if p.velocity_window < 1 or min(p.velocity_scale, p.risk_radius, p.adaptive_radius) <= 0:
        parser.error('Motion windows and scales must be positive')
    if p.num_epochs < 1 or p.epochs_per_run < 0:
        parser.error('num_epochs must be positive and epochs_per_run nonnegative')
    if p.train_model is None:
        p.train_model = experiment_name(p)
    p.save_dir = os.path.join(p.save_base_dir, str(p.test_set))
    p.model_dir = os.path.join(p.save_dir, p.train_model)
    p.config = os.path.join(p.model_dir, 'config_' + p.phase + '.yaml')
    # print(p.seq_length) # 5 + 8 = 13
    if not load_arg(p, parser):
        save_arg(p)
    args = load_arg(p, parser)
    # Resolve seeds after loading configuration, then record the actual CLI overrides.
    prepare_seed(args.seed)
    save_arg(args)
    print(args.num_pred)
    if args.using_cuda:
        torch.cuda.set_device(args.gpu)
    return args

if __name__ == '__main__':
    args = parse_run_args()
    processor = Processor(args)
    if args.phase=='test':
        processor.playtest()
    else:
        processor.playtrain()
