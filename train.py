import argparse
import datetime
import logging
import os
import os.path as osp
import time

import torch
import tqdm
from torch.utils.data import DataLoader

from evaluate import evaluate
from face_dataset import FaceMask
from logger import setup_logger
from loss import OhemCELoss
from model import BiSeNet
from optimizer import Optimizer

respth = './res'
if not osp.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def train(args):
    setup_logger(respth)

    # dataset
    n_classes = 19
    batch_size = 8
    n_workers = 0
    cropsize = [448, 448]

    ds = FaceMask(args.dataset_dir, cropsize=cropsize, mode='train')
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=True)

    # model
    ignore_idx = -100
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.train()
    score_thres = 0.7
    n_min = batch_size * cropsize[0] * cropsize[1] // 16
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
        model=net,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power)

    # train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in tqdm.tqdm(range(max_iter), 'Train'):
        try:
            im, lb = next(diter)
            if not im.size()[0] == batch_size:
                raise StopIteration
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16, out32 = net(im)
        lossp = LossP(out, lb)
        loss2 = Loss2(out16, lb)
        loss3 = Loss3(out32, lb)
        loss = lossp + loss2 + loss3
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        #  print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it=it + 1,
                max_it=max_iter,
                lr=lr,
                loss=loss_avg,
                time=t_intv,
                eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed
        if (it + 1) % 5000 == 0:
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            torch.save(state, './res/{}_iter.pth'.format(it))
            os.makedirs(args.dataset_dir + 'training', exist_ok=True)
            evaluate(dspth=args.dataset_dir + 'training', cp='{}_iter.pth'.format(it))

    #  dump the final model
    save_pth = osp.join(respth, 'model_final_diss.pth')
    # net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
