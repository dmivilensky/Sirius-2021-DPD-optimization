'''
This is the code version of this https://arxiv.org/pdf/2106.02720.pdf optimizer
Written by Peters Egor Alexandrovich
 under the guidance of Pasechnyuk Dmitry Arkadievich and Gasnikov Alexander Vladimirovich
'''


import torch
import copy
from torch.optim import optimizer


class AccMbSGD(optimizer.Optimizer):
    def __init__(self, params, lr=0.9):  # the best lr was around 2 btw
        default = dict(lr=lr)
        self.B = None  # defining first values for Beta=None as it'll be changed at the first step
        self.t = 0  # and defining t as 0 as it'll be changed after each step
        super(AccMbSGD, self).__init__(params, default)

    def __setstate__(self, state):
        super(AccMbSGD, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            gamma = group["lr"]

            if self.B is None:  # as Beta is defined as None at __init__()
                self.B = 0  # it has to be changed to a number such as 0
                for p in group["params"]:
                    self.B += p.norm(2)  # then it is added all model weights in 2-norm
                self.B = self.B.sqrt()  # and then it's square rooted

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                if "w_ag" not in param_state:  # it's necessary to put w_ag and w at the start
                    param_state["w_ag"] = copy.deepcopy(p.data)  # to keep them between epochs
                if "w" not in param_state:
                    param_state["w"] = copy.deepcopy(p.data)
                beta = 1 + self.t / 6  # the optimizer body itself

                p.data.mul_(0)
                p.data.add_(1 - 1 / beta, param_state["w_ag"])
                p.data.add_(1 / beta, param_state["w"])

                param_state["w"].add_(-gamma, p.grad.data)
                param_state["w"].mul_(min(1, self.B / param_state["w"].norm(2)))

                param_state["w_ag"].mul_(1 - 1 / beta)
                param_state["w_ag"].add_(1 / beta, param_state["w"])

            self.t += 1  # changing t after learning
        return loss
