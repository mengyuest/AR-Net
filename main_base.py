import warnings

warnings.filterwarnings("ignore")

import os
import sys
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models_ada import TSN_Ada
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map, Recorder

from tensorboardX import SummaryWriter
from ops.my_logger import Logger

from ops.sal_rank_loss import cal_sal_rank_loss

from ops.net_flops_table import get_gflops_params, feat_dim_dict
from ops.utils import get_mobv2_new_sd

from os.path import join as ospj


def load_to_sd(model_dict, model_path, module_name, fc_name, resolution, apple_to_apple=False):
    if ".pth" in model_path:
        print("done loading\t%s\t(res:%3d) from\t%s" % ("%-25s" % module_name, resolution, model_path))
        sd = torch.load(model_path)['state_dict']
        new_version_detected = False
        for k in sd:
            if "lite_backbone.features.1.conv.4." in k:
                new_version_detected = True
                break
        if new_version_detected:
            sd = get_mobv2_new_sd(sd, reverse=True)

        if apple_to_apple:
            del_keys = []
            if args.remove_all_base_0:
                for key in sd:
                    if "module.base_model_list.0" in key or "new_fc_list.0" in key or "linear." in key:
                        del_keys.append(key)

            if args.no_weights_from_linear:
                for key in sd:
                    if "linear." in key:
                        del_keys.append(key)

            for key in list(set(del_keys)):
                del sd[key]

            return sd

        replace_dict = []
        nowhere_ks = []
        notfind_ks = []

        for k, v in sd.items():  # TODO(yue) base_model->base_model_list.i
            new_k = k.replace("base_model", module_name)
            new_k = new_k.replace("new_fc", fc_name)
            if new_k in model_dict:
                replace_dict.append((k, new_k))
            else:
                nowhere_ks.append(k)
        for new_k, v in model_dict.items():
            if module_name in new_k:
                k = new_k.replace(module_name, "base_model")
                if k not in sd:
                    notfind_ks.append(k)
            if fc_name in new_k:
                k = new_k.replace(fc_name, "new_fc")
                if k not in sd:
                    notfind_ks.append(k)
        if len(nowhere_ks) != 0:
            print("Vars not in ada network, but are in pretrained weights\n" + ("\n%s NEW  " % module_name).join(
                nowhere_ks))
        if len(notfind_ks) != 0:
            print("Vars not in pretrained weights, but are needed in ada network\n" + ("\n%s LACK " % module_name).join(
                notfind_ks))
        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)

        if "lite_backbone" in module_name:
            # TODO not loading new_fc in this case, because we are using hidden_dim
            if args.frame_independent == False:
                del sd["module.lite_fc.weight"]
                del sd["module.lite_fc.bias"]
        return {k: v for k, v in sd.items() if k in model_dict}
    else:
        print("skip loading\t%s\t(res:%3d) from\t%s" % ("%-25s" % module_name, resolution, model_path))
        return {}


def main():
    t_start = time.time()
    global args, best_prec1, num_class, use_ada_framework  # , model

    set_random_seed(args.random_seed)
    use_ada_framework = args.ada_reso_skip and args.offline_lstm_last == False and args.offline_lstm_all == False and args.real_scsampler == False

    if args.ablation:
        logger = None
    else:
        if not test_mode:
            logger = Logger()
            sys.stdout = logger
        else:
            logger = None

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.data_dir)

    if args.ada_reso_skip:
        if len(args.ada_crop_list) == 0:
            args.ada_crop_list = [1 for _ in args.reso_list]

    if use_ada_framework:
        init_gflops_table()

    model = TSN_Ada(num_class, args.num_segments,
                    base_model=args.arch,
                    consensus_type=args.consensus_type,
                    dropout=args.dropout,
                    partial_bn=not args.no_partialbn,
                    pretrain=args.pretrain,
                    fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                    args=args)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    # TODO(yue) freeze some params in the policy + lstm layers
    if args.freeze_policy:
        for name, param in model.module.named_parameters():
            if "lite_fc" in name or "lite_backbone" in name or "rnn" in name or "linear" in name:
                param.requires_grad = False

    if args.freeze_backbone:
        for name, param in model.module.named_parameters():
            if "base_model" in name:
                param.requires_grad = False
    if len(args.frozen_list) > 0:
        for name, param in model.module.named_parameters():
            for keyword in args.frozen_list:
                if keyword[0] == "*":
                    if keyword[-1] == "*":  # TODO middle
                        if keyword[1:-1] in name:
                            param.requires_grad = False
                            print(keyword, "->", name, "frozen")
                    else:  # TODO suffix
                        if name.endswith(keyword[1:]):
                            param.requires_grad = False
                            print(keyword, "->", name, "frozen")
                elif keyword[-1] == "*":  # TODO prefix
                    if name.startswith(keyword[:-1]):
                        param.requires_grad = False
                        print(keyword, "->", name, "frozen")
                else:  # TODO exact word
                    if name == keyword:
                        param.requires_grad = False
                        print(keyword, "->", name, "frozen")
        print("=" * 80)
        for name, param in model.module.named_parameters():
            print(param.requires_grad, "\t", name)

        print("=" * 80)
        for name, param in model.module.named_parameters():
            print(param.requires_grad, "\t", name)

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    # TODO(yue) ada_model loading process
    if args.ada_reso_skip:
        if test_mode:
            print("Test mode load from pretrained model")
            the_model_path = args.test_from
            if ".pth.tar" not in the_model_path:
                the_model_path = ospj(the_model_path, "models", "ckpt.best.pth.tar")
            model_dict = model.state_dict()
            sd = load_to_sd(model_dict, the_model_path, "foo", "bar", -1, apple_to_apple=True)
            model_dict.update(sd)
            model.load_state_dict(model_dict)
        elif args.base_pretrained_from != "":
            print("Adaptively load from pretrained whole")
            model_dict = model.state_dict()
            sd = load_to_sd(model_dict, args.base_pretrained_from, "foo", "bar", -1, apple_to_apple=True)

            model_dict.update(sd)
            model.load_state_dict(model_dict)

        elif len(args.model_paths) != 0:
            print("Adaptively load from model_path_list")
            model_dict = model.state_dict()
            # TODO(yue) policy net
            sd = load_to_sd(model_dict, args.policy_path, "lite_backbone", "lite_fc",
                            args.reso_list[args.policy_input_offset])
            model_dict.update(sd)
            # TODO(yue) backbones
            for i, tmp_path in enumerate(args.model_paths):
                base_model_index = i
                new_i = i

                sd = load_to_sd(model_dict, tmp_path, "base_model_list.%d" % base_model_index, "new_fc_list.%d" % new_i,
                                args.reso_list[i])
                model_dict.update(sd)
            model.load_state_dict(model_dict)
    else:
        if test_mode:
            the_model_path = args.test_from
            if ".pth.tar" not in the_model_path:
                the_model_path = ospj(the_model_path, "models", "ckpt.best.pth.tar")
            model_dict = model.state_dict()
            sd = load_to_sd(model_dict, the_model_path, "foo", "bar", -1, apple_to_apple=True)
            model_dict.update(sd)
            model.load_state_dict(model_dict)

    if args.ada_reso_skip == False and args.base_pretrained_from != "":
        print("Baseline: load from pretrained model")
        model_dict = model.state_dict()
        sd = load_to_sd(model_dict, args.base_pretrained_from, "base_model", "new_fc", 224)

        if args.ignore_new_fc_weight:
            print("@ IGNORE NEW FC WEIGHT !!!")
            del sd["module.new_fc.weight"]
            del sd["module.new_fc.bias"]

        model_dict.update(sd)
        model.load_state_dict(model_dict)

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)
    data_length = 1
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]), dense_sample=args.dense_sample,
                   dataset=args.dataset,
                   partial_fcvid_eval=args.partial_fcvid_eval,
                   partial_ratio=args.partial_ratio,
                   ada_reso_skip=args.ada_reso_skip,
                   reso_list=args.reso_list,
                   random_crop=args.random_crop,
                   center_crop=args.center_crop,
                   ada_crop_list=args.ada_crop_list,
                   rescale_to=args.rescale_to,
                   policy_input_offset=args.policy_input_offset,
                   save_meta=args.save_meta),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]), dense_sample=args.dense_sample,
                   dataset=args.dataset,
                   partial_fcvid_eval=args.partial_fcvid_eval,
                   partial_ratio=args.partial_ratio,
                   ada_reso_skip=args.ada_reso_skip,
                   reso_list=args.reso_list,
                   random_crop=args.random_crop,
                   center_crop=args.center_crop,
                   ada_crop_list=args.ada_crop_list,
                   rescale_to=args.rescale_to,
                   policy_input_offset=args.policy_input_offset,
                   save_meta=args.save_meta
                   ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    if not test_mode:
        exp_full_path = setup_log_directory(logger, args.log_dir, args.exp_header)
    else:
        exp_full_path = None

    if not args.ablation:
        if not test_mode:
            with open(os.path.join(exp_full_path, 'args.txt'), 'w') as f:
                f.write(str(args))
            tf_writer = SummaryWriter(log_dir=exp_full_path)
        else:
            tf_writer = None
    else:
        tf_writer = None

    # TODO(yue)
    map_record = Recorder()
    mmap_record = Recorder()
    prec_record = Recorder()
    best_train_usage_str = None
    best_val_usage_str = None

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if not args.skip_training:
            set_random_seed(args.random_seed + epoch)
            adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
            train_usage_str = train(train_loader, model, criterion, optimizer, epoch, logger, exp_full_path, tf_writer)
        else:
            train_usage_str = "No training usage stats (Eval Mode)"

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            set_random_seed(args.random_seed)
            mAP, mmAP, prec1, val_usage_str, val_gflops = validate(val_loader, model, criterion, epoch, logger,
                                                                   exp_full_path, tf_writer)

            # remember best prec@1 and save checkpoint
            map_record.update(mAP)
            mmap_record.update(mmAP)
            prec_record.update(prec1)

            if mmap_record.is_current_best():
                best_train_usage_str = train_usage_str
                best_val_usage_str = val_usage_str

            print('Best mAP: %.3f (epoch=%d)\t\tBest mmAP: %.3f(epoch=%d)\t\tBest Prec@1: %.3f (epoch=%d)' % (
                map_record.best_val, map_record.best_at,
                mmap_record.best_val, mmap_record.best_at,
                prec_record.best_val, prec_record.best_at))

            if args.skip_training:
                break

            if (not args.ablation) and (not test_mode):
                tf_writer.add_scalar('acc/test_top1_best', prec_record.best_val, epoch)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': prec_record.best_val,
                }, mmap_record.is_current_best(), exp_full_path)

    if use_ada_framework and not test_mode:
        print("Best train usage:")
        print(best_train_usage_str)
        print()
        print("Best val usage:")
        print(best_val_usage_str)

    print("Finished in %.4f seconds\n" % (time.time() - t_start))


def set_random_seed(the_seed):
    if args.random_seed >= 0:
        np.random.seed(the_seed)
        torch.manual_seed(the_seed)


def init_gflops_table():
    global gflops_table
    gflops_table = {}
    seg_len = -1

    for i, backbone in enumerate(args.backbone_list):
        gflops_table[backbone + str(args.reso_list[i])] = \
            get_gflops_params(backbone, args.reso_list[i], num_class, seg_len)[0]
    gflops_table["policy"] = \
        get_gflops_params(args.policy_backbone, args.reso_list[args.policy_input_offset], num_class, seg_len)[0]
    gflops_table["lstm"] = 2 * (feat_dim_dict[args.policy_backbone] ** 2) / 1000000000

    print("gflops_table: ")
    for k in gflops_table:
        print("%-20s: %.4f GFLOPS" % (k, gflops_table[k]))


def get_gflops_t_tt_vector():
    gflops_vec = []
    t_vec = []
    tt_vec = []

    for i, backbone in enumerate(args.backbone_list):
        if all([arch_name not in backbone for arch_name in ["resnet", "mobilenet", "efficientnet", "res3d", "csn"]]):
            exit("We can only handle resnet/mobilenet/efficientnet/res3d/csn as backbone, when computing FLOPS")

        for crop_i in range(args.ada_crop_list[i]):
            the_flops = gflops_table[backbone + str(args.reso_list[i])]
            gflops_vec.append(the_flops)
            t_vec.append(1.)
            tt_vec.append(1.)

    if args.policy_also_backbone:
        gflops_vec.append(0)
        t_vec.append(1.)
        tt_vec.append(1.)

    for i, _ in enumerate(args.skip_list):
        t_vec.append(1. if args.skip_list[i] == 1 else 1. / args.skip_list[i])
        tt_vec.append(0)
        gflops_vec.append(0)

    return gflops_vec, t_vec, tt_vec


def cal_eff(r):
    each_losses = []
    # TODO r N * T * (#reso+#policy+#skips)
    gflops_vec, t_vec, tt_vec = get_gflops_t_tt_vector()
    t_vec = torch.tensor(t_vec).cuda()
    if args.use_gflops_loss:
        r_loss = torch.tensor(gflops_vec).cuda()
    else:
        r_loss = torch.tensor([4., 2., 1., 0.5, 0.25, 0.125, 0.0625, 0.03125]).cuda()[:r.shape[2]]

    loss = torch.sum(torch.mean(r, dim=[0, 1]) * r_loss)
    each_losses.append(loss.detach().cpu().item())

    # TODO(yue) uniform loss
    if args.uniform_loss_weight > 1e-5:
        if_policy_backbone = 1 if args.policy_also_backbone else 0
        num_pred = len(args.backbone_list)
        policy_dim = num_pred + if_policy_backbone + len(args.skip_list)

        reso_skip_vec = torch.zeros(policy_dim).cuda()

        # TODO
        offset = 0
        # TODO reso/ada_crops
        for b_i in range(num_pred):
            interval = args.ada_crop_list[b_i]
            reso_skip_vec[b_i] += torch.sum(r[:, :, offset:offset + interval])
            offset = offset + interval

        # TODO mobilenet + skips
        for b_i in range(num_pred, reso_skip_vec.shape[0]):
            reso_skip_vec[b_i] = torch.sum(r[:, :, b_i])

        reso_skip_vec = reso_skip_vec / torch.sum(reso_skip_vec)
        if args.uniform_cross_entropy:  # TODO cross-entropy+ logN
            uniform_loss = torch.sum(
                torch.tensor([x * torch.log(torch.clamp_min(x, 1e-6)) for x in reso_skip_vec])) + torch.log(
                torch.tensor(1.0 * len(reso_skip_vec)))
            uniform_loss = uniform_loss * args.uniform_loss_weight
        else:  # TODO L2 norm
            usage_bias = reso_skip_vec - torch.mean(reso_skip_vec)
            uniform_loss = torch.norm(usage_bias, p=2) * args.uniform_loss_weight
        loss = loss + uniform_loss
        each_losses.append(uniform_loss.detach().cpu().item())

    # TODO(yue) high-reso punish loss
    if args.head_loss_weight > 1e-5:
        head_usage = torch.mean(r[:, :, 0])
        usage_threshold = 0.2
        head_loss = (head_usage - usage_threshold) * (head_usage - usage_threshold) * args.head_loss_weight
        loss = loss + head_loss
        each_losses.append(head_loss.detach().cpu().item())

    # TODO(yue) frames loss
    if args.frames_loss_weight > 1e-5:
        num_frames = torch.mean(torch.mean(r, dim=[0, 1]) * t_vec)
        frames_loss = num_frames * num_frames * args.frames_loss_weight
        loss = loss + frames_loss
        each_losses.append(frames_loss.detach().cpu().item())

    return loss, each_losses


def reverse_onehot(a):
    try:
        return np.array([np.where(r > 0.5)[0][0] for r in a])
    except Exception as e:
        print("error stack:", e)
        print(a)
        for i, r in enumerate(a):
            print(i, r)
        return None


def get_criterion_loss(criterion, output, target):
    return criterion(output, target[:, 0])


def kl_categorical(p_logit, q_logit):
    import torch.nn.functional as F
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                         - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)


def compute_acc_eff_loss_with_weights(acc_loss, eff_loss, each_losses, epoch):
    if epoch > args.eff_loss_after:
        acc_weight = args.accuracy_weight
        eff_weight = args.efficency_weight
    else:
        acc_weight = 1.0
        eff_weight = 0.0
    return acc_loss * acc_weight, eff_loss * eff_weight, [x * eff_weight for x in each_losses]


def compute_every_losses(r, acc_loss, epoch):
    eff_loss, each_losses = cal_eff(r)
    acc_loss, eff_loss, each_losses = compute_acc_eff_loss_with_weights(acc_loss, eff_loss, each_losses, epoch)
    return acc_loss, eff_loss, each_losses


def elastic_list_print(l, limit=8):
    if isinstance(l, str):
        return l

    limit = min(limit, len(l))
    l_output = "[%s," % (",".join([str(x) for x in l[:limit // 2]]))
    if l.shape[0] > limit:
        l_output += "..."
    l_output += "%s]" % (",".join([str(x) for x in l[-limit // 2:]]))
    return l_output


def compute_exp_decay_tau(epoch):
    return args.init_tau * np.exp(args.exp_decay_factor * epoch)


def get_policy_usage_str(r_list, reso_dim):
    gflops_vec, t_vec, tt_vec = get_gflops_t_tt_vector()
    printed_str = ""
    rs = np.concatenate(r_list, axis=0)

    tmp_cnt = [np.sum(rs[:, :, iii] == 1) for iii in range(rs.shape[2])]

    if args.all_policy:
        tmp_total_cnt = tmp_cnt[0]
    else:
        tmp_total_cnt = sum(tmp_cnt)

    gflops = 0
    avg_frame_ratio = 0
    avg_pred_ratio = 0

    used_model_list = []
    reso_list = []

    for i in range(len(args.backbone_list)):
        used_model_list += [args.backbone_list[i]] * args.ada_crop_list[i]
        reso_list += [args.reso_list[i]] * args.ada_crop_list[i]

    for action_i in range(rs.shape[2]):
        if args.policy_also_backbone and action_i == reso_dim - 1:
            action_str = "m0(%s %dx%d)" % (
                args.policy_backbone, args.reso_list[args.policy_input_offset],
                args.reso_list[args.policy_input_offset])
        elif action_i < reso_dim:
            action_str = "r%d(%7s %dx%d)" % (
                action_i, used_model_list[action_i], reso_list[action_i], reso_list[action_i])
        else:
            action_str = "s%d (skip %d frames)" % (action_i - reso_dim, args.skip_list[action_i - reso_dim])

        usage_ratio = tmp_cnt[action_i] / tmp_total_cnt
        printed_str += "%-22s: %6d (%.2f%%)\n" % (action_str, tmp_cnt[action_i], 100 * usage_ratio)

        gflops += usage_ratio * gflops_vec[action_i]
        avg_frame_ratio += usage_ratio * t_vec[action_i]
        avg_pred_ratio += usage_ratio * tt_vec[action_i]

    num_clips = args.num_segments
    gflops += (gflops_table["policy"] + gflops_table["lstm"]) * avg_frame_ratio
    printed_str += "GFLOPS: %.6f  AVG_FRAMES: %.3f  NUM_PREDS: %.3f" % (
        gflops, avg_frame_ratio * args.num_segments, avg_pred_ratio * num_clips)
    return printed_str, gflops


def extra_each_loss_str(each_terms):
    loss_str_list = ["gf"]
    s = ""
    if args.uniform_loss_weight > 1e-5:
        loss_str_list.append("u")
    if args.head_loss_weight > 1e-5:
        loss_str_list.append("h")
    if args.frames_loss_weight > 1e-5:
        loss_str_list.append("f")
    for i in range(len(loss_str_list)):
        s += " %s:(%.4f)" % (loss_str_list[i], each_terms[i].avg)
    return s


def get_current_temperature(num_epoch):
    if args.exp_decay:
        tau = compute_exp_decay_tau(num_epoch)
    else:
        tau = args.init_tau
    return tau


def get_average_meters(number):
    return [AverageMeter() for _ in range(number)]


def train(train_loader, model, criterion, optimizer, epoch, logger, exp_full_path, tf_writer):
    batch_time, data_time, losses, top1, top5 = get_average_meters(5)
    tau = 0
    if use_ada_framework:
        tau = get_current_temperature(epoch)
        alosses, elosses = get_average_meters(2)
        each_terms = get_average_meters(NUM_LOSSES)
        r_list = []

    meta_offset = -2 if args.save_meta else 0

    model.module.partialBN(not args.no_partialbn)

    # switch to train mode
    model.train()

    end = time.time()
    print("#%s# lr:%.4f\ttau:%.4f" % (
        args.exp_header, optimizer.param_groups[-1]['lr'] * 0.1, tau if use_ada_framework else 0))

    for i, input_tuple in enumerate(train_loader):

        data_time.update(time.time() - end)  # TODO(yue) measure data loading time

        target = input_tuple[-1].cuda()
        target_var = torch.autograd.Variable(target)

        input = input_tuple[0]
        if args.ada_reso_skip:
            input_var_list = [torch.autograd.Variable(input_item) for input_item in input_tuple[:-1 + meta_offset]]

            if args.real_scsampler:
                output, r, real_pred, lite_pred = model(input=input_var_list, tau=tau)
                if args.sal_rank_loss:
                    acc_loss = cal_sal_rank_loss(real_pred, lite_pred, target_var)
                else:
                    acc_loss = get_criterion_loss(criterion, lite_pred.mean(dim=1), target_var)
            else:
                if args.use_reinforce:
                    output, r, r_log_prob, base_outs = model(input=input_var_list, tau=tau)
                    acc_loss = get_criterion_loss(criterion, output, target_var)
                else:
                    output, r, feat_outs, base_outs = model(input=input_var_list, tau=tau)
                    acc_loss = get_criterion_loss(criterion, output, target_var)

            if use_ada_framework:
                acc_loss, eff_loss, each_losses = compute_every_losses(r, acc_loss, epoch)

                if args.use_reinforce and not args.freeze_policy:
                    if args.separated:
                        acc_loss_items = []
                        eff_loss_items = []

                        for b_i in range(output.shape[0]):
                            acc_loss_item = get_criterion_loss(criterion, output[b_i:b_i + 1], target_var[b_i:b_i + 1])
                            acc_loss_item, eff_loss_item, each_losses_item = compute_every_losses(r[b_i:b_i + 1],
                                                                                                  acc_loss_item, epoch)

                            acc_loss_items.append(acc_loss_item)
                            eff_loss_items.append(eff_loss_item)

                        if args.no_baseline:
                            b_acc = 0
                            b_eff = 0
                        else:
                            b_acc = sum(acc_loss_items) / len(acc_loss_items)
                            b_eff = sum(eff_loss_items) / len(eff_loss_items)

                        log_p = torch.mean(r_log_prob, dim=1)

                        acc_loss = sum(acc_loss_items) / len(acc_loss_items)
                        eff_loss = sum(eff_loss_items) / len(eff_loss_items)

                        if args.detach_reward:
                            acc_loss_vec = (torch.stack(acc_loss_items) - b_acc).detach()
                            eff_loss_vec = (torch.stack(eff_loss_items) - b_eff).detach()
                        else:
                            acc_loss_vec = (torch.stack(acc_loss_items) - b_acc)
                            eff_loss_vec = (torch.stack(eff_loss_items) - b_eff)

                        intended_acc_loss = torch.mean(log_p * acc_loss_vec)
                        intended_eff_loss = torch.mean(log_p * eff_loss_vec)

                        each_losses = [0 * each_l for each_l in each_losses]

                    else:
                        sum_log_prob = torch.sum(r_log_prob) / r_log_prob.shape[0] / r_log_prob.shape[1]
                        acc_loss = - sum_log_prob * acc_loss
                        eff_loss = - sum_log_prob * eff_loss
                        each_losses = [-sum_log_prob * each_l for each_l in each_losses]

                    intended_loss = intended_acc_loss + intended_eff_loss

                alosses.update(acc_loss.item(), input.size(0))
                elosses.update(eff_loss.item(), input.size(0))

                for l_i, each_loss in enumerate(each_losses):
                    each_terms[l_i].update(each_loss, input.size(0))
                loss = acc_loss + eff_loss
            else:
                loss = acc_loss
        else:
            input_var = torch.autograd.Variable(input)
            output = model(input=[input_var])
            loss = get_criterion_loss(criterion, output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        if args.use_reinforce and not args.freeze_policy:
            intended_loss.backward()
        else:
            loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if use_ada_framework:
            r_list.append(r.detach().cpu().numpy())

        if i % args.print_freq == 0:
            print_output = ('Epoch:[{0:02d}][{1:03d}/{2:03d}] '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            '{data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))  # TODO

            if use_ada_framework:
                roh_r = reverse_onehot(r[-1, :, :].detach().cpu().numpy())
                print_output += ' a {aloss.val:.4f} ({aloss.avg:.4f}) e {eloss.val:.4f} ({eloss.avg:.4f}) r {r}'.format(
                    aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r)
                )
                print_output += extra_each_loss_str(each_terms)
            if args.show_pred:
                print_output += elastic_list_print(output[-1, :].detach().cpu().numpy())
            print(print_output)

    if use_ada_framework:
        usage_str, gflops = get_policy_usage_str(r_list, model.module.reso_dim)
        print(usage_str)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

    return usage_str if use_ada_framework else None


def validate(val_loader, model, criterion, epoch, logger, exp_full_path, tf_writer=None):
    batch_time, losses, top1, top5 = get_average_meters(4)
    tau = 0
    # TODO(yue)
    all_results = []
    all_targets = []
    all_all_preds = []

    i_dont_need_bb = True

    if use_ada_framework:
        tau = get_current_temperature(epoch)
        alosses, elosses = get_average_meters(2)

        iter_list = args.backbone_list

        if not i_dont_need_bb:
            all_bb_results = [[] for _ in range(len(iter_list))]
            if args.policy_also_backbone:
                all_bb_results.append([])

        each_terms = get_average_meters(NUM_LOSSES)
        r_list = []
        if args.save_meta:
            name_list = []
            indices_list = []

    meta_offset = -2 if args.save_meta else 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):

            target = input_tuple[-1].cuda()
            input = input_tuple[0]

            # compute output
            if args.ada_reso_skip:
                if args.real_scsampler:
                    output, r, real_pred, lite_pred = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                    if args.sal_rank_loss:
                        acc_loss = cal_sal_rank_loss(real_pred, lite_pred, target)
                    else:
                        acc_loss = get_criterion_loss(criterion, lite_pred.mean(dim=1), target)
                else:
                    if args.save_meta and args.save_all_preds:
                        output, r, all_preds = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                        acc_loss = get_criterion_loss(criterion, output, target)
                    else:
                        if args.use_reinforce:
                            output, r, r_log_prob, base_outs = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                            acc_loss = get_criterion_loss(criterion, output, target)
                        else:
                            output, r, feat_outs, base_outs = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                            acc_loss = get_criterion_loss(criterion, output, target)
                if use_ada_framework:
                    acc_loss, eff_loss, each_losses = compute_every_losses(r, acc_loss, epoch)

                    if args.use_reinforce and not args.freeze_policy:
                        if args.separated:
                            acc_loss_items = []
                            eff_loss_items = []
                            for b_i in range(output.shape[0]):
                                acc_loss_item = get_criterion_loss(criterion, output[b_i:b_i + 1],
                                                                   target[b_i:b_i + 1])
                                acc_loss_item, eff_loss_item, each_losses_item = compute_every_losses(r[b_i:b_i + 1],
                                                                                                      acc_loss_item,
                                                                                                      epoch)
                                acc_loss_items.append(acc_loss_item)
                                eff_loss_items.append(eff_loss_item)

                            if args.no_baseline:
                                b_acc = 0
                                b_eff = 0
                            else:
                                b_acc = sum(acc_loss_items) / len(acc_loss_items)
                                b_eff = sum(eff_loss_items) / len(eff_loss_items)

                            log_p = torch.mean(r_log_prob, dim=1)
                            acc_loss = 0
                            eff_loss = 0
                            for b_i in range(len(acc_loss_items)):
                                acc_loss += -log_p[b_i] * (acc_loss_items[b_i] - b_acc)
                                eff_loss += -log_p[b_i] * (eff_loss_items[b_i] - b_eff)
                            acc_loss = acc_loss / len(acc_loss_items)
                            eff_loss = eff_loss / len(eff_loss_items)
                            each_losses = [0 * each_l for each_l in each_losses]
                        else:
                            sum_log_prob = torch.sum(r_log_prob) / r_log_prob.shape[0] / r_log_prob.shape[1]
                            acc_loss = - sum_log_prob * acc_loss
                            eff_loss = - sum_log_prob * eff_loss
                            each_losses = [-sum_log_prob * each_l for each_l in each_losses]

                    alosses.update(acc_loss.item(), input.size(0))
                    elosses.update(eff_loss.item(), input.size(0))
                    for l_i, each_loss in enumerate(each_losses):
                        each_terms[l_i].update(each_loss, input.size(0))
                    loss = acc_loss + eff_loss
                else:
                    loss = acc_loss
            else:
                output = model(input=[input])
                loss = get_criterion_loss(criterion, output, target)

            # TODO(yue)
            all_results.append(output)
            all_targets.append(target)

            if not i_dont_need_bb:
                for bb_i in range(len(all_bb_results)):
                    all_bb_results[bb_i].append(base_outs[:, bb_i])

            if args.save_meta and args.save_all_preds:
                all_all_preds.append(all_preds)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if use_ada_framework:
                r_list.append(r.cpu().numpy())
                if args.save_meta:
                    name_list += input_tuple[-3]
                    indices_list.append(input_tuple[-2])

            if i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                if use_ada_framework:
                    roh_r = reverse_onehot(r[-1, :, :].cpu().numpy())

                    print_output += ' a {aloss.val:.4f} ({aloss.avg:.4f}) e {eloss.val:.4f} ({eloss.avg:.4f}) r {r}'.format(
                        aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r)
                    )

                    print_output += extra_each_loss_str(each_terms)

                print(print_output)

    mAP, _ = cal_map(torch.cat(all_results, 0).cpu(),
                     torch.cat(all_targets, 0)[:, 0:1].cpu())  # TODO(yue) single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())  # TODO(yue)  multi-label mAP
    print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses))

    if not i_dont_need_bb:
        bbmmaps = []
        bbprec1s = []
        all_targets_cpu = torch.cat(all_targets, 0).cpu()
        for bb_i in range(len(all_bb_results)):
            bb_results_cpu = torch.mean(torch.cat(all_bb_results[bb_i], 0), dim=1).cpu()
            bb_i_mmAP, _ = cal_map(bb_results_cpu, all_targets_cpu)  # TODO(yue)  multi-label mAP
            bbmmaps.append(bb_i_mmAP)

            bbprec1, = accuracy(bb_results_cpu, all_targets_cpu[:, 0], topk=(1,))
            bbprec1s.append(bbprec1)

        print("bbmmAP: " + " ".join(["{0:.3f}".format(bb_i_mmAP) for bb_i_mmAP in bbmmaps]))
        print("bb_Acc: " + " ".join(["{0:.3f}".format(bbprec1) for bbprec1 in bbprec1s]))
    gflops = 0

    if use_ada_framework:
        usage_str, gflops = get_policy_usage_str(r_list, model.module.reso_dim)
        print(usage_str)

        if args.save_meta:  # TODO save name, label, r, result

            npa = np.concatenate(r_list)
            npb = np.stack(name_list)
            npc = torch.cat(all_results).cpu().numpy()
            npd = torch.cat(all_targets).cpu().numpy()
            if args.save_all_preds:
                npe = torch.cat(all_all_preds).cpu().numpy()
            else:
                npe = np.zeros(1)

            npf = torch.cat(indices_list).cpu().numpy()

            np.savez("%s/meta-val-%s.npy" % (exp_full_path, logger._timestr),
                     rs=npa, names=npb, results=npc, targets=npd, all_preds=npe, indices=npf)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return mAP, mmAP, top1.avg, usage_str if use_ada_framework else None, gflops


def save_checkpoint(state, is_best, exp_full_path):
    if is_best:
        torch.save(state, '%s/models/ckpt.best.pth.tar' % (exp_full_path))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def setup_log_directory(logger, log_dir, exp_header):
    if args.ablation:
        return None

    exp_full_name = "g%s_%s" % (logger._timestr, exp_header)
    exp_full_path = ospj(log_dir, exp_full_name)
    os.makedirs(exp_full_path)
    os.makedirs(ospj(exp_full_path, "models"))
    logger.create_log(exp_full_path, test_mode, args.num_segments, args.batch_size, args.top_k)
    return exp_full_path


if __name__ == '__main__':
    best_prec1 = 0
    num_class = -1
    use_ada_framework = False
    NUM_LOSSES = 10
    gflops_table = {}
    args = parser.parse_args()
    test_mode = (args.test_from != "")

    if test_mode:  # TODO test mode
        print("======== TEST MODE ========")
        args.skip_training = True

    main()
