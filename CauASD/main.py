# -*- coding: gbk -*-
from config import *
from dataset import DataSet
from logger import Log
import math
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
from torch.autograd import Function

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

torch.cuda.set_device(1)
setup_seed(0)

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, l):
        ctx.l = l
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反转梯度，并乘以 λ
        return grad_output.neg() * ctx.l, None


class GradientReversalLayer(nn.Module):
    def __init__(self, l=1.0):
        super().__init__()
        self.l = l

    def forward(self, x):
        return GradReverse.apply(x, self.l)

class RandomTemporalIntervention(nn.Module):
    def __init__(self, min_speed=0.5, max_speed=2.0):
        super().__init__()
        self.min_speed = min_speed
        self.max_speed = max_speed

    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.shape
        device = x.device

        speed_factors = torch.rand(N, device=device) * (self.max_speed - self.min_speed) + self.min_speed
        t = torch.arange(T, device=device).float().view(1, -1)  # (1, T)
        t = t.expand(N, T)  # (N, T)
        t_new = t / speed_factors.view(N, 1)
        t_new = t_new.clamp(0, T - 1)

        x_flat = x.permute(0, 1, 3, 4, 2).reshape(N, -1, T)  # (N, C*V*M, T)

        left = torch.floor(t_new).long()
        right = torch.ceil(t_new).long()
        w = (t_new - left.float()).unsqueeze(1)

        ch = x_flat.size(1)
        left_idx = left.unsqueeze(1).expand(-1, ch, -1)
        right_idx = right.unsqueeze(1).expand(-1, ch, -1)

        left_vals = torch.gather(x_flat, 2, left_idx)
        right_vals = torch.gather(x_flat, 2, right_idx)
        x_warp_flat = (1 - w) * left_vals + w * right_vals
        x_warp = x_warp_flat.view(N, C, V, M, T).permute(0, 1, 4, 2, 3)

        return x_warp, speed_factors

class SpeedPredictor(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def compute_speed_from_skeleton(data):
    # data: (N, C, T, V, M)
    with torch.no_grad():
        diff = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]  # (N, C, T-1, V, M)
        mag = torch.norm(diff, dim=1)  # (N, T-1, V, M)
        speed = mag.mean(dim=(1, 2, 3))
    return speed  # (N,)

# %%
class Processor:

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

    def adjust_learning_rate(self, optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch=15,
                             loss_mode='step', step=[50, 80]):
        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        elif loss_mode == 'cos':
            lr = lr_min + (lr_max - lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        elif loss_mode == 'step':
            lr = lr_max * (0.1 ** np.sum(current_epoch >= np.array(step)))
        else:
            raise Exception('Please check loss_mode!')

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            # if i == 0:
            #     param_group['lr'] = lr * 0.1
            # else:
            #     param_group['lr'] = lr

    def layernorm(self, feature):
        num = feature.shape[0]
        mean = torch.mean(feature, dim=1).reshape(num, -1)
        var = torch.var(feature, dim=1).reshape(num, -1)
        out = (feature - mean) / torch.sqrt(var)

        return out

    @ex.capture
    def load_model(self, in_channels, hidden_channels, hidden_dim,
                   dropout, graph_args, edge_importance_weighting, alpha_grl ,visual_size, language_size, weight_path, loss_type,
                   fix_encoder, finetune):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                             hidden_dim=hidden_dim, dropout=dropout,
                             graph_args=graph_args,
                             edge_importance_weighting=edge_importance_weighting,
                             )
        self.encoder = self.encoder.cuda()
        self.adapter = Linear().cuda()
        self.grl = GradientReversalLayer(l=alpha_grl).cuda()
        self.speed_pred = SpeedPredictor(in_dim=language_size).cuda()
        if loss_type == "kl" or loss_type == "klv2" or loss_type == "kl+cosface" or loss_type == "kl+sphereface" or "kl+margin":
            self.loss = KLLoss().cuda()
        else:
            raise Exception('loss_type Error!')
        self.logit_scale = self.adapter.get_logit_scale()
        self.logit_scale_v2 = self.adapter.get_logit_scale_v2()

        if fix_encoder or finetune:
            self.load_weights(self.encoder, weight_path)

    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        self.optimizer = torch.optim.SGD([
            {'params': self.encoder.parameters()},
            {'params': self.adapter.parameters()},
            {'params': self.speed_pred.parameters()},
        ],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=False
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100)

    @ex.capture
    def optimize(self, epoch_num, DA):  # print -> log.info
        self.log.info("main track")
        for epoch in range(epoch_num):
            self.train_epoch(epoch)
            with torch.no_grad():
                self.test_epoch(epoch=epoch)
            self.log.info("epoch [{}] train loss: {}".format(epoch, self.dim_loss))
            self.log.info("epoch [{}] discriminator train loss: {}".format(epoch, self.discri_loss))
            self.log.info("epoch [{}] mainadversarial train loss: {}".format(epoch, self.mainadversarial_loss))
            self.log.info("epoch [{}] test acc: {}".format(epoch, self.test_acc))
            self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch, self.best_acc))

            if DA:
                self.log.info("epoch [{}] DA test acc: {}".format(epoch, self.test_aug_acc))
                self.log.info("epoch [{}] gets the best DA acc: {}".format(self.best_aug_epoch, self.best_aug_acc))
            # if epoch > 5:
            #     self.log.info("epoch [{}] test acc: {}".format(epoch,self.test_acc))
            #     self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
            # else:
            #     self.log.info("epoch [{}] : warm up epoch.".format(epoch))

    @ex.capture
    def train_epoch(self, epoch, lr, loss_mode, step, loss_type, beta_tci, lambda_bsdm, intervene_k, fix_encoder, batch_size):
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr, warmup_epoch=5,
                                  loss_mode=loss_mode, step=step)
        running_loss = []
        loader = self.data_loader['train']

        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            # print(data.shape) #128,3,50,25,2
            # label = label.type(torch.LongTensor).cuda()
            label_g = gen_label(label)
            label = label.type(torch.LongTensor).cuda()

            # print(label) # int
            seen_language = self.full_language[label]
            # print(seen_language.shape)
            speed_real = compute_speed_from_skeleton(data)
            self.encoder.train()
            self.adapter.train()
            self.speed_pred.train()
            feat = self.encoder(data)
            skleton_feat = self.adapter(feat)

            if loss_type == "kl":
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                # ground_truth = gen_label_from_text_sim(seen_language)
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                L_align = (loss_skls + loss_texts) / 2
            else:
                raise Exception('loss_type Error!')
            temporal_aug = RandomTemporalIntervention(0.5, 2.0)
            total_intervention_loss = 0
            total_intervention_loss_text = 0
            total_feat_cons = 0
            aug_speed_list = []
            aug_feat_list = []

            for _ in range(intervene_k):
                aug_data, aug_speed = temporal_aug(data)
                aug_data = aug_data.float().cuda()
                feat_aug = self.encoder(aug_data)
                skleton_feat_aug = self.adapter(feat_aug)
                skleton_feat_aug = skleton_feat_aug.view(skleton_feat_aug.size(0), -1)
                skleton_feat_aug = F.normalize(skleton_feat_aug, dim=-1)

                logits_aug, logits_text_aug = create_logits(
                    skleton_feat_aug, seen_language, self.logit_scale, exp=True
                )

                L_int = F.kl_div(
                    F.log_softmax(logits_aug, dim=-1),
                    F.softmax(logits_per_skl.detach(), dim=-1),
                    reduction="batchmean"
                )
                L_int_text = F.kl_div(
                    F.log_softmax(logits_text_aug, dim=-1),
                    F.softmax(logits_per_text.detach(), dim=-1),
                    reduction="batchmean"
                )

                cos_sim = F.cosine_similarity(skleton_feat_aug, skleton_feat.detach(), dim=-1)
                L_feat = 1 - cos_sim.mean()
                total_intervention_loss += L_int
                total_intervention_loss_text += L_int_text
                total_feat_cons += L_feat

                aug_speed_list.append(compute_speed_from_skeleton(aug_data))
                aug_feat_list.append(skleton_feat_aug)

            total_intervention_loss /= intervene_k
            total_intervention_loss_text /= intervene_k
            total_feat_cons /= intervene_k

            speed_targets = [speed_real] + aug_speed_list
            speed_targets_all = torch.cat([s.view(-1) for s in speed_targets], dim=0)

            feat_list_for_speed = [skleton_feat] + aug_feat_list
            feat_speed_all = torch.cat(feat_list_for_speed, dim=0)
            # === GRL + 速度预测器 ===
            feat_grl = self.grl(feat_speed_all)
            speed_pred = self.speed_pred(feat_grl)
            L_speed = F.mse_loss(speed_pred, speed_targets_all.to(speed_pred.device))

            loss = (
                    L_align
                    + beta_tci * total_intervention_loss
                    + beta_tci * total_intervention_loss_text
                    + lambda_bsdm * L_speed
            )

            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()

    @ex.capture
    def test_epoch(self, unseen_label, epoch, DA, support_factor):
        self.encoder.eval()
        self.adapter.eval()
        self.speed_pred.eval()
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
            # inference
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
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            self.save_model()
            # y_true = np.array(y_true)
            # y_pred = np.array(y_pred)
            # np.save("y_true_3.npy",y_true)
            # np.save("y_pred_3.npy",y_pred)
            # print("save ok!")
        self.test_acc = acc

        if DA:
            ent_all = torch.cat(ent_list)
            feat_all = torch.cat(feat_list)
            old_pred_all = torch.cat(old_pred_list)
            z_list = []
            for i in range(len(unseen_label)):
                mask = old_pred_all == i
                class_support_set = feat_all[mask]
                class_ent = ent_all[mask]
                class_len = class_ent.shape[0]
                if int(class_len * support_factor) < 1:
                    z = self.full_language[unseen_label[i:i + 1]]
                else:
                    _, indices = torch.topk(-class_ent, int(class_len * support_factor))
                    z = torch.mean(class_support_set[indices], dim=0, keepdim=True)
                z_list.append(z)

            z_tensor = torch.cat(z_list)
            aug_acc_list = []
            for data, label in tqdm(loader):
                # y_t = label.numpy().tolist()
                # y_true += y_t

                data = data.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                unseen_language = z_tensor
                # inference
                feature = self.encoder(data)
                feature = self.adapter(feature)
                # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
                acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)

                # y_p = pred.cpu().numpy().tolist()
                # y_pred += y_p
                aug_acc_list.append(acc_batch)
            aug_acc = torch.tensor(aug_acc_list).mean()
            if aug_acc > self.best_aug_acc:
                self.best_aug_acc = aug_acc
                self.best_aug_epoch = epoch
            self.test_aug_acc = aug_acc

    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    @ex.capture
    def save_model(self, save_path):
        torch.save({'encoder': self.encoder.state_dict(), 'adapter': self.adapter.state_dict()}, save_path)

    def start(self):
        self.initialize()
        self.optimize()
        # self.save_model()

# %%
@ex.automain
def main(track):
    p = Processor()
    p.start()
