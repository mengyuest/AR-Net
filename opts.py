import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str)
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=3)
# parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
# parser.add_argument('--loss_type', type=str, default="nll",
#                    choices=['nll'])
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')  # TODO(changed from 120 to 50)
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",  # TODO(changed from [50,100] to [20,40])
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  # TODO(changed from 5e-4 to 1e-4)
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')  # TODO(changed from None to 20)
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')  # TODO(changed from 5 to 1)

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--root_log', type=str, default='logs')
parser.add_argument('--root_model', type=str, default='checkpoint')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')

# TODO(yue) ADAPTIVE RESEARCH HYPER-PARAMETERS
parser.add_argument('--exp_header', default="default", type=str, help='experiment header')
parser.add_argument('--rescale_to', default=224, type=int)

# TODO(yue) adaptive resolution and skipping (hardcoded version)
parser.add_argument('--ada_reso_skip', action='store_true', help='adaptively select scale and choose to skip')
parser.add_argument('--reso_list', default=[224], type=int, nargs='+', help="list of resolutions")
parser.add_argument('--skip_list', default=[], type=int, nargs='+', help="list of frames to skip")
parser.add_argument('--backbone_list', default=[], type=str, nargs='+', help="backbones for diff resos")
parser.add_argument('--shared_backbone', action='store_true', help="share same backbone weight")
parser.add_argument('--accuracy_weight', default=1., type=float)
parser.add_argument('--efficency_weight', default=0., type=float)
parser.add_argument('--show_pred', action='store_true')

# TODO(yue) multi-label cases (for activity-net-v1.3)
# Always provides (single + multi) mAPs. Difference is only in training
parser.add_argument('--loss_type', type=str, default="nll", choices=['nll', 'bce'])

# TODO(yue) for policy network:
parser.add_argument('--policy_backbone', default='mobilenet_v2', type=str, help="backbones for policy network")
parser.add_argument('--policy_input_offset', default=0, type=int, help="select which scale for policy network")
parser.add_argument('--hidden_dim', default=512, type=int, help="dimension for hidden state and cell state")
parser.add_argument('--offline_lstm_last', action='store_true', help="just using LSTM(last one), no policy")
parser.add_argument('--offline_lstm_all', action='store_true', help="just using LSTM(all average), no policy")
parser.add_argument('--random_policy', action='store_true', help="just using random policy there")
parser.add_argument('--all_policy', action='store_true', help="just using all feat there")
parser.add_argument('--eff_loss_after', default=-1, type=int, help="use eff loss after X epochs")

parser.add_argument('--save_freq', default=10, type=int, help="freq to save network model weight")  # TODO(yue)
parser.add_argument('--model_paths', default=[], type=str, nargs="+", help='path to load models for backbones')
parser.add_argument('--policy_path', default="", type=str, help="path of the policy network")
# TODO(yue) maybe we want to use ImageNet pretrain or not, depending on the resolution

# TODO(yue) annealing
parser.add_argument('--exp_decay', action='store_true', help="type of annealing")
parser.add_argument('--init_tau', default=5.0, type=float, help="annealing init temperature")
parser.add_argument('--exp_decay_factor', default=-0.045, type=float, help="exp decay factor per epoch")

# TODO(yue) small tweak
parser.add_argument('--policy_from_scratch', action='store_true', help="policy network without pretraining")
parser.add_argument('--frozen_list', default=[], type=str, nargs="+", help='list of frozen part')
parser.add_argument('--policy_also_backbone', action='store_true', help="use policy as the last backbone")
parser.add_argument('--uniform_loss_weight', type=float, default=1e-6, help="loss to constraints all uses equal")
parser.add_argument('--lite_mode', action='store_true')

# TODO(yue) try different losses for efficiency terms
parser.add_argument('--use_gflops_loss', action='store_true')  # TODO(yue) use flops as loss assignment
parser.add_argument('--head_loss_weight', type=float, default=1e-6)  # TODO(yue) punish to the high resolution selection
parser.add_argument('--frames_loss_weight', type=float, default=1e-6)  # TODO(yue) use num_frames as a loss assignment

# TODO(yue) finetuning and testing
parser.add_argument('--base_pretrained_from', type=str, default='', help='for base model pretrained path')
parser.add_argument('--skip_training', action='store_true')  # TODO(yue) just doing eval
parser.add_argument('--freeze_policy', action='store_true')  # TODO(yue) fix the policy

# TODO(yue) reproducibility
parser.add_argument('--random_seed', type=int, default=1007)

# TODO(yue) for FCVID or datasets where eval is too heavy
parser.add_argument('--partial_fcvid_eval', action='store_true')
parser.add_argument('--partial_ratio', type=float, default=0.2)

# TODO(yue) crop
parser.add_argument('--center_crop', action='store_true')
parser.add_argument('--random_crop', action='store_true')

# TODO(yue) oracle scsampler (from ListenToLook ideas)
parser.add_argument('--consensus_type', type=str, default='avg')  # TODO can also use scsampler!
parser.add_argument('--top_k', type=int, default=10)  # TODO can also use scsampler!

# TODO(yue) real SCSampler (we also use --top_k to select frames, and use consensus type=='scsampler')
parser.add_argument('--real_scsampler', action='store_true')
parser.add_argument('--sal_rank_loss', action='store_true')
parser.add_argument('--frame_independent', action='store_true')  # TODO use tsn in models_ada
parser.add_argument('--freeze_backbone', action='store_true')

# TODO(yue) reproducibility
parser.add_argument('--test_from', type=str, default="")

# TODO(yue) adaptive-cropping (only 1, 5, 9)
parser.add_argument('--ada_crop_list', default=[], type=int, nargs="+", help='num of anchor points per scaling')

# TODO(yue) visualizations
parser.add_argument('--save_meta', action='store_true')
parser.add_argument('--ablation', action='store_true')
parser.add_argument('--remove_all_base_0', action='store_true')
parser.add_argument('--save_all_preds', action='store_true')

# TODO(yue) RL
parser.add_argument('--default_signal', default=0, type=int)
parser.add_argument('--uniform_cross_entropy', action='store_true')
parser.add_argument('--ignore_new_fc_weight', action='store_true')
parser.add_argument('--no_weights_from_linear', action='store_true')
parser.add_argument('--uno_reso', action='store_true')

parser.add_argument('--use_reinforce', action='store_true')
parser.add_argument('--separated', action='store_true')
parser.add_argument('--no_baseline', action='store_true')
parser.add_argument('--detach_reward', action='store_true')
parser.add_argument('--negative_loss', action='store_true')

# TODO(yue) Exps/data paths
parser.add_argument('--data_dir', type=str, default="../../datasets/activity-net-v1.3")
parser.add_argument('--log_dir', type=str, default="../../logs_tsm")
