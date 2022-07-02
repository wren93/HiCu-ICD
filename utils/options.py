import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--MODEL_DIR', type=str, default='./models')
parser.add_argument('--DATA_DIR', type=str, default='./data')
parser.add_argument('--MIMIC_3_DIR', type=str, default='./data/mimic3')
parser.add_argument('--MIMIC_2_DIR', type=str, default='./data/mimic2')

parser.add_argument("--data_path", type=str,
                    default='./data/mimic3/train_full.csv')
parser.add_argument("--vocab", type=str, default='./data/mimic3/vocab.csv')
parser.add_argument("--Y", type=str, default='full', choices=['full', '50'])
parser.add_argument("--version", type=str,
                    choices=['mimic2', 'mimic3'], default='mimic3')
parser.add_argument("--MAX_LENGTH", type=int, default=4096)

# model
parser.add_argument("--model", type=str, choices=[
                    'MultiResCNN', 'longformer', 'RACReader', 'LAAT'], default='MultiResCNN')
parser.add_argument("--decoder", type=str, choices=['HierarchicalHyperbolic', 'Hierarchical', 'LAATHierarchicalHyperbolic', 'LAATHierarchical',
                                                   'CodeTitle', 'RandomlyInitialized', 'LAATDecoder'], default='HierarchicalHyperbolic')
parser.add_argument("--filter_size", type=str, default="3,5,9,15,19,25",
                    help="Conv layer filter size for MultiResCNN and RAC model. For MultiResCNN, this is a list of integers seperated by comma; for RAC, this is a single integer number")
parser.add_argument("--num_filter_maps", type=int, default=50)
parser.add_argument("--conv_layer", type=int, default=1)
parser.add_argument("--embed_file", type=str,
                    default='./data/mimic3/processed_full_100.embed')
parser.add_argument("--hyperbolic_dim", type=int, default=50)
parser.add_argument("--test_model", type=str, default=None)
parser.add_argument("--use_ext_emb", action="store_const",
                    const=True, default=False)
parser.add_argument('--cat_hyperbolic', action="store_const",
                    const=True, default=False)
parser.add_argument("--loss", type=str, choices=['BCE', 'ASL', 'ASLO'], default='BCE')
parser.add_argument("--asl_config", type=str, default='0,0,0')
parser.add_argument("--asl_reduction", type=str, choices=['mean', 'sum'], default='sum')

# training
parser.add_argument("--n_epochs", type=str, default="2,3,5,10,500")
parser.add_argument("--depth", type=int, default=5)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--criterion", type=str, default='prec_at_8',
                    choices=['prec_at_8', 'f1_micro', 'prec_at_5'])
parser.add_argument("--gpu", type=str, default='0',
                    help='-1 if not using gpu, use comma to separate multiple gpus')
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--tune_wordemb", action="store_const",
                    const=True, default=True)
parser.add_argument('--random_seed', type=int, default=1,
                    help='0 if randomly initialize the model, other if fix the seed')

parser.add_argument("--thres", type=float, default=0.5)

# longformer
parser.add_argument("--longformer_dir", type=str, default='')

# RAC encoder
parser.add_argument("--reader_conv_num", type=int, default=2)
parser.add_argument("--reader_trans_num", type=int, default=4)
parser.add_argument("--trans_ff_dim", type=int, default=1024)

# RAC decoder
parser.add_argument("--num_code_title_tokens", type=int, default=36)
parser.add_argument("--code_title_filter_size", type=int, default=9)

# LAAT
parser.add_argument("--lstm_hidden_dim", type=int, default=512)
parser.add_argument("--attn_dim", type=int, default=512)
parser.add_argument("--scheduler", type=float, default=0.9)
parser.add_argument("--scheduler_patience", type=int, default=5)

args = parser.parse_args()
command = ' '.join(['python'] + sys.argv)
args.command = command

# gpu settings
args.gpu_list = [int(idx) for idx in args.gpu.split(',')]
args.gpu_list = [i for i in range(
    len(args.gpu_list))] if args.gpu_list[0] >= 0 else [-1]
