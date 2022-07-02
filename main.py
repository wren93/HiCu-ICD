import os
import time
import random
import numpy as np
import torch
import csv
import sys
from collections import defaultdict
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AdamW
from gensim.models.poincare import PoincareModel

from utils.utils import (
    load_lookups,
    prepare_instance,
    MyDataset,
    my_collate,
    my_collate_longformer,
    early_stop,
    save_everything,
    prepare_instance_longformer,
    prepare_code_title
)
from utils.options import args
from utils.models import pick_model
from utils.train_test import train, test


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    print(args)

    maxInt = sys.maxsize
    while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    # load vocab and other lookups
    print("loading lookups...")
    dicts = load_lookups(args) # load lookup table for tokens and icd codes

    if args.decoder.find("CodeTitle") != -1:
        dicts['c2title'] = prepare_code_title(dicts, args, args.num_code_title_tokens)

    if args.decoder.find("Hyperbolic") != -1:
        print("Training hyperbolic embeddings...")
        hierarchy = dicts['hierarchy_dist']
        # train poincare (hyperbolic) embeddings
        relations = set()
        for k, v in hierarchy[4].items():
            relations.add(('root', v[0]))
            for i in range(4):
                relations.add(tuple(v[i:i+2]))
        relations = list(relations)
        poincare = PoincareModel(relations, args.hyperbolic_dim, negative=10)
        poincare.train(epochs=50)
        dicts['poincare_embeddings'] = poincare.kv
    
    if args.decoder == "CodeTitle" or args.decoder == "RandomlyInitialized" or args.decoder == "LAATDecoder":
        args.depth = 1

    model = pick_model(args, dicts)
    print(model)

    if not args.test_model:
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    else:
        optimizer = None

    if args.model.find("longformer") != -1:
        prepare_instance_func = prepare_instance_longformer
    else:
        prepare_instance_func = prepare_instance

    train_instances = prepare_instance_func(dicts, args.data_path, args, args.MAX_LENGTH)
    print("train_instances {}".format(len(train_instances)))
    if args.version != 'mimic2':
        dev_instances = prepare_instance_func(dicts, args.data_path.replace('train','dev'), args, args.MAX_LENGTH)
        print("dev_instances {}".format(len(dev_instances)))
    else:
        dev_instances = None
    test_instances = prepare_instance_func(dicts, args.data_path.replace('train','test'), args, args.MAX_LENGTH)
    print("test_instances {}".format(len(test_instances)))

    if args.model.find("longformer") != -1:
        collate_func = my_collate_longformer
    else:
        collate_func = my_collate

    train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=True)
    if args.version != 'mimic2':
        dev_loader = DataLoader(MyDataset(dev_instances), 1, shuffle=False, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=True)
    else:
        dev_loader = None
    test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=True)

    scheduler = None
    if args.model.find("LAAT") != -1 and not args.test_model:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler)
    
    if not args.test_model and args.model.find("longformer") != -1:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    test_only = args.test_model is not None

    start_depth = 5 - args.depth
    cur_depth = 4 if test_only else start_depth

    epochs = [int(epoch) for epoch in args.n_epochs.split(',')]
    print(f"Total epochs at each level: {epochs}")

    while cur_depth < 5:
        metrics_hist = defaultdict(lambda: [])
        metrics_hist_te = defaultdict(lambda: [])
        metrics_hist_tr = defaultdict(lambda: [])
        break_loop = False
        if not test_only:
            print("Training model at depth {}:".format(cur_depth))
            if cur_depth != 0:
                if isinstance(model, torch.nn.DataParallel):
                    model.module.decoder.change_depth(cur_depth)
                else:
                    model.decoder.change_depth(cur_depth)
        for epoch in range(epochs[cur_depth]):
            if epoch == 0 and cur_depth == start_depth and not args.test_model:
                model_dir = os.path.join(args.MODEL_DIR, '_'.join([args.model, args.decoder, time.strftime('%b_%d_%H_%M_%S', time.localtime())]))
                os.makedirs(model_dir)
            elif args.test_model:
                model_dir = os.path.dirname(os.path.abspath(args.test_model))

            if not test_only and not break_loop:
                epoch_start = time.time()
                losses = train(args, model, optimizer, scheduler, epoch, args.gpu_list, train_loader, cur_depth)
                loss = np.mean(losses)
                epoch_finish = time.time()
                print("epoch finish in %.2fs, loss: %.4f" % (epoch_finish - epoch_start, loss))
            else:
                loss = np.nan

            fold = 'test' if args.version == 'mimic2' else 'dev'
            dev_instances = test_instances if args.version == 'mimic2' else dev_instances
            dev_loader = test_loader if args.version == 'mimic2' else dev_loader
            if epoch == epochs[cur_depth] - 1:
                print("last epoch: testing on dev and test sets")
                break_loop = True

            # test on dev
            evaluation_start = time.time()
            metrics = test(args, model, args.data_path, fold, args.gpu_list, dicts, dev_loader, cur_depth)
            evaluation_finish = time.time()
            print("evaluation finish in %.2fs" % (evaluation_finish - evaluation_start))
            if test_only or break_loop or epoch == epochs[cur_depth] - 1:
                metrics_te = test(args, model, args.data_path, "test", args.gpu_list, dicts, test_loader, cur_depth)
            else:
                metrics_te = defaultdict(float)
            metrics_tr = {'loss': loss}
            metrics_all = (metrics, metrics_te, metrics_tr)

            for name in metrics_all[0].keys():
                metrics_hist[name].append(metrics_all[0][name])
            for name in metrics_all[1].keys():
                metrics_hist_te[name].append(metrics_all[1][name])
            for name in metrics_all[2].keys():
                metrics_hist_tr[name].append(metrics_all[2][name])
            metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

            save_everything(args, metrics_hist_all, model, model_dir, None, args.criterion, test_only)

            sys.stdout.flush()

            if test_only or break_loop:
                break

            if args.criterion in metrics_hist.keys():
                if early_stop(metrics_hist, args.criterion, args.patience):
                    #stop training, do tests on test and train sets, and then stop the script
                    print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                    break_loop = True
                    args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                    tmp = args.depth
                    args.depth = 5 - cur_depth
                    model = pick_model(args, dicts)
                    args.depth = tmp

            if scheduler is not None and args.criterion in metrics_hist.keys():
                if early_stop(metrics_hist, args.criterion, args.scheduler_patience):
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print(f"{args.criterion} hasn't improved in {args.scheduler_patience} epochs, reduce learning rate to {param_group['lr']}")

        cur_depth += 1
