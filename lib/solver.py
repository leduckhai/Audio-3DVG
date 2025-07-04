import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import importlib

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd(), "utils"))  # HACK add the lib folder
import lib

importlib.reload(lib)
from lib.config import CONF
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from lib.scheduler_helper import BNMomentumScheduler

from utils.eta import decode_eta

ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_ref_loss: {train_ref_loss}
[sco.] train_ref_acc: {train_ref_acc}
[sco.] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[val]   val_loss: {val_loss}
[val]   val_ref_acc: {val_ref_acc}
[val]   val_iou_rate_0.25: {val_iou_rate_25}, val_iou_rate_0.5: {val_iou_rate_5}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] ref_loss: {ref_loss}
[sco.] ref_acc: {ref_acc}
[sco.] iou_rate_0.25: {iou_rate_25}, iou_rate_0.5: {iou_rate_5}
"""


class Solver():
    def __init__(self, model, config, dataloader, optimizer, stamp, val_step=10,
                 lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None):

        self.epoch = 0  # set in __call__
        self.verbose = 0  # set in __call__

        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step

        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_decay_rate = bn_decay_rate

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "ref_loss": float("inf"),
            "ref_acc": -float("inf"),
            "iou_rate_0.25": -float("inf"),
            "iou_rate_0.5": -float("inf")
        }

        # log
        self.init_log()

        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}  # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        if lr_decay_step and lr_decay_rate:
            if isinstance(lr_decay_step, list):
                self.lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate ** (int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)
        else:
            self.bn_scheduler = None

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * self.val_step

        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))
                
                # feed
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last_{}.pth").format(epoch_id))

                print("evaluating...")
                # self.init_log()
                # val
                self._feed(self.dataloader["val"], "val", epoch_id)

                # update lr scheduler
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                    self._log("update learning rate --> {}\n".format(self.lr_scheduler.get_last_lr()))

                # update bn scheduler
                if self.bn_scheduler:
                    self.bn_scheduler.step()
                    self._log("update batch normalization momentum --> {}\n".format(
                        self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))

            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)

        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        data_dict = get_loss(
            data_dict=data_dict,
            config=self.config,
        )

        # dump
        self._running_log["ref_loss"] = data_dict["ref_loss"]
        self._running_log["loss"] = data_dict["loss"]

    def _eval(self, data_dict):
        data_dict = get_eval(
            data_dict=data_dict,
            config=self.config,
        )

        # dump
        self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])
        self._running_log['ref_iou'] = data_dict['ref_iou']

    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)

        # change dataloader
        dataloader = dataloader if phase == "train" else tqdm(dataloader)
        fetch_time_start = time.time()
        sum_label = 0
        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                if key in ['object_cat', 'point_min', 'point_max', 'mlm_label', # REMOVE lang_len
                           'ref_center_label', 'ref_size_residual_label']:
                    data_dict[key] = data_dict[key].cuda()

            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                "ref_loss": 0,
                # acc
                "ref_acc": 0,
                "iou_rate_0.25": 0,
                "iou_rate_0.5": 0
            }

            # load
            self.log[phase]["fetch"].append(time.time() - fetch_time_start)

            # debug only
            # with torch.autograd.set_detect_anomaly(True):
            # forward
            start = time.time()
            data_dict = self._forward(data_dict)
            self._compute_loss(data_dict)
            self.log[phase]["forward"].append(time.time() - start)

            # backward
            if phase == "train":
                start = time.time()
                self._backward()
                self.log[phase]["backward"].append(time.time() - start)

            # eval
            start = time.time()
            self._eval(data_dict)
            self.log[phase]["eval"].append(time.time() - start)

            # record log
            self.log[phase]["loss"].append(self._running_log["loss"].item())
            self.log[phase]["ref_loss"].append(self._running_log["ref_loss"].item())
            self.log[phase]["ref_acc"].append(self._running_log["ref_acc"])
            self.log[phase]['ref_iou'] += self._running_log['ref_iou']

            ious = self.log[phase]['ref_iou']
            self.log[phase]['iou_rate_0.25'] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
            self.log[phase]['iou_rate_0.5'] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]

            # report
            if phase == "train":
                iter_time = self.log[phase]["fetch"][-1]
                iter_time += self.log[phase]["forward"][-1]
                iter_time += self.log[phase]["backward"][-1]
                iter_time += self.log[phase]["eval"][-1]
                self.log[phase]["iter_time"].append(iter_time)
                if (self._global_iter_id + 1) % self.verbose == 0:
                    self._train_report(epoch_id)
                    # dump log
                    self._dump_log("train")
                    self.init_log()

                self._global_iter_id += 1
            fetch_time_start = time.time()
        

        # check best
        if phase == "val":
            ious = self.log[phase]['ref_iou']
            self.log[phase]['iou_rate_0.25'] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
            self.log[phase]['iou_rate_0.5'] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]

            self._dump_log("val")
            self._epoch_report(epoch_id)

            cur_criterion = "iou_rate_0.25"
            cur_best = self.log[phase][cur_criterion]
            if cur_best > self.best[cur_criterion]:
                self._log("best {} achieved: {}".format(cur_criterion, cur_best))
                self.best["epoch"] = epoch_id + 1
                self.best["loss"] = np.mean(self.log[phase]["loss"])
                self.best["ref_loss"] = np.mean(self.log[phase]["ref_loss"])
                self.best["ref_acc"] = np.mean(self.log[phase]["ref_acc"])
                self.best["iou_rate_0.25"] = self.log[phase]['iou_rate_0.25']
                self.best["iou_rate_0.5"] = self.log[phase]['iou_rate_0.5']

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "best_model.pth"))

    def _dump_log(self, phase):
        log = {
            "loss": ["loss", "ref_loss"],
            "score": ["ref_acc"]
        }
        for key in log:
            for item in log[key]:
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(key, item),
                    np.mean([v for v in self.log[phase][item]]),
                    self._global_iter_id
                )

        self._log_writer[phase].add_scalar(
            "{}/{}".format("score", 'iou_rate_0.25'),
            self.log[phase]['iou_rate_0.25'],
            self._global_iter_id
        )
        self._log_writer[phase].add_scalar(
            "{}/{}".format("score", 'iou_rate_0.5'),
            self.log[phase]['iou_rate_0.5'],
            self._global_iter_id
        )


    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(
                os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * np.ceil(self._total_iter["train"] / self.val_step) * mean_est_val_time
        eta = decode_eta(eta_sec)


    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id + 1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
            val_ref_loss=round(np.mean([v for v in self.log["val"]["ref_loss"]]), 5),
            val_ref_acc=round(np.mean([v for v in self.log["val"]["ref_acc"]]), 5),
            val_iou_rate_25=round(self.log['val']['iou_rate_0.25'], 5),
            val_iou_rate_5=round(self.log['val']['iou_rate_0.5'], 5),
        )
        self._log(epoch_report)

    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            ref_loss=round(self.best["ref_loss"], 5),
            ref_acc=round(self.best["ref_acc"], 5),
            iou_rate_25=round(self.best["iou_rate_0.25"], 5),
            iou_rate_5=round(self.best["iou_rate_0.5"], 5),
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)

    def init_log(self):
        # contains all necessary info for all phases
        self.log = {
            phase: {
                # info
                "forward": [],
                "backward": [],
                "eval": [],
                "fetch": [],
                "iter_time": [],
                # loss (float, not torch.cuda.FloatTensor)
                "loss": [],
                "ref_loss": [],
                # scores (float, not torch.cuda.FloatTensor)
                "ref_acc": [],
                'ref_iou': [],
                "iou_rate_0.25": [],
                "iou_rate_0.5": []
            } for phase in ["train", "val"]
        }
