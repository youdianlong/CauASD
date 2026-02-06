from config import *
from dataset import DataSet
from logger import Log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import pi, cos
from tqdm import tqdm
from module.gcn.st_gcn import Model
from module.adapter import Adapter, Linear
from KLLoss import KLLoss, KDLoss
from tool import gen_label, create_logits, get_acc, create_sim_matrix, gen_label_from_text_sim, get_m_theta, get_acc_v2

torch.cuda.set_device(1)

def _reshape_to_flat_time(x):
    # x: (N,C,T,V,M) -> (N, C*V*M, T)
    N, C, T, V, M = x.shape
    return x.permute(0, 1, 3, 4, 2).reshape(N, C * V * M, T)
def _reshape_from_flat_time(x_flat, N, C, T, V, M):
    # x_flat: (N, C*V*M, T) -> (N,C,T,V,M)
    return x_flat.reshape(N, C, V, M, T).permute(0, 1, 4, 2, 3)
def time_scale(x, min_speed=0.5, max_speed=0.5):
    N, C, T, V, M = x.shape
    x_flat = _reshape_to_flat_time(x)  # (N, C*V*M, T)
    device = x.device
    speed_factors = torch.rand(N, device=device) * (max_speed - min_speed) + min_speed
    out = torch.zeros_like(x_flat)
    t = torch.arange(T, device=device).float()
    for i in range(N):
        sf = float(speed_factors[i].item())
        t_new = t / sf
        t_new = t_new.clamp(0, T - 1)
        left = t_new.floor().long()
        right = t_new.ceil().long()
        w = (t_new - left).unsqueeze(0)  # (1, T)
        xi = x_flat[i]  # (C*V*M, T)
        # gather left/right
        left_vals = xi[:, left]
        right_vals = xi[:, right]
        out[i] = (1 - w) * left_vals + w * right_vals

    return _reshape_from_flat_time(out, N, C, T, V, M)

class testpt:
    @ex.capture
    def load_data(self, train_list, train_label, test_list, test_label, batch_size, language_path):
        self.dataset = dict()
        self.data_loader = dict()
        self.best_epoch = -1
        self.best_acc = -1
        self.dim_loss = -1
        self.discri_loss = -1
        self.mainadversarial_loss = -1
        self.dis_loss = -1
        self.test_acc = -1
        self.test_aug_acc = -1
        self.best_aug_acc = -1
        self.best_aug_epoch = -1

        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = self.full_language.cuda()
        self.dataset['train'] = DataSet(train_list, train_label)
        self.dataset['test'] = DataSet(test_list, test_label)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            drop_last=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=64,
            num_workers=16,
            shuffle=False)

    def load_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict)


    @ex.capture
    def load_model(self, in_channels, hidden_channels, hidden_dim,
                   dropout, graph_args, edge_importance_weighting, visual_size, language_size, weight_path, loss_type,
                   fix_encoder, finetune):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                             hidden_dim=hidden_dim, dropout=dropout,
                             graph_args=graph_args,
                             edge_importance_weighting=edge_importance_weighting,
                             )
        self.encoder = self.encoder.cuda()
        self.adapter = Linear().cuda()

        self.proj_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 256)).cuda()
        if loss_type == "kl" or loss_type == "klv2" or loss_type == "kl+cosface" or loss_type == "kl+sphereface" or "kl+margin":
            self.loss = KLLoss().cuda()
        else:
            raise Exception('loss_type Error!')
        self.logit_scale = self.adapter.get_logit_scale()
        self.logit_scale_v2 = self.adapter.get_logit_scale_v2()


        #best acc checkpoint
        checkpoint = torch.load("./output/model/split_1_kl_DA_des_support_factor0.9_lr0.05.pt")


        self.encoder.load_state_dict(checkpoint['encoder'])
        self.adapter.load_state_dict(checkpoint['adapter'])

        if fix_encoder or finetune:
            self.load_weights(self.encoder, weight_path)

        self.encoder.eval()
        self.adapter.eval()

    @ex.capture
    def optimize(self, epoch_num, DA):  # print -> log.info
        self.log.info("main track")

        with torch.no_grad():
            self.test_epoch()
        self.log.info("epoch [{}] test acc: {}".format(1, self.test_acc))

    @ex.capture
    def test_epoch(self, unseen_label, DA, support_factor):
        self.encoder.eval()
        self.adapter.eval()

        loader = self.data_loader['test']
        y_true = []
        y_pred = []
        acc_list = []
        ent_list = []
        feat_list = []
        old_pred_list = []
        for data, label in tqdm(loader):

            # y_t = label.numpy().tolist()
            # y_true += y_t

            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            unseen_language = self.full_language[unseen_label]
            data=time_scale(data)
            # inference
            with torch.no_grad():
                feature = self.encoder(data)
                feature = self.adapter(feature)
            if DA:
                # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
                acc_batch, pred, old_pred, ent, feat = get_acc_v2(feature, unseen_language, unseen_label, label)
                ent_list.append(ent)
                feat_list.append(feat)
                old_pred_list.append(old_pred)
            else:
                acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)

            # y_p = pred.cpu().numpy().tolist()
            # y_pred += y_p

            acc_list.append(acc_batch)

        acc_list = torch.tensor(acc_list)

        acc = acc_list.mean()

        self.test_acc = acc


    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    def start(self):
        self.initialize()
        self.optimize()
        # self.save_model()


@ex.automain
def main():
    p = testpt();
    p.start();

