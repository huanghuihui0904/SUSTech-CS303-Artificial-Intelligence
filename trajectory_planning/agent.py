import copy
import os
import torch.nn as nn
import torch.optim as optim


import os.path as osp
import torch
# from torch import vmap
from typing import Tuple

import time


class BlobModel(nn.Module):
    def __init__(self):
        """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 8.
        """
        super().__init__()
        self.flatten=nn.Flatten()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(), # <- does our dataset require non-linear layers? (try uncommenting and see if the results change)
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(), # <- does our dataset require non-linear layers? (try uncommenting and see if the results change)
            nn.Linear(in_features=64, out_features=10), # how many classes are there?



        )

    def forward(self, x):
        x=self.flatten(x)
        logists=self.linear_layer_stack(x)
        return logists

# Create an instance of BlobModel and send it to the target device

class Agent:



    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """
        # TODO: prepare your agent here

        # self.model = torch.load(os.path.abspath('./model2.pth'))
        device =  "cpu"
        self.model_4 = BlobModel().to(device)
        modelpath=osp.join(osp.dirname(__file__),"model8.pth")
        self.model_dict =self.model_4.load_state_dict(torch.load(modelpath))
        self.agentScores = []


    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters required to fire a projectile.

        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets.
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """
        assert len(target_pos) == len(target_features)
        start = time.time()


        # TODO: compute the firing speed and angle that would give the best score.
        # Example: return a random configuration
        # 基于target_features得到target_cls
        target_cls=None
        self.model_4.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = self.model_4(target_features)
            target_cls = torch.softmax(test_logits, dim=1).argmax(dim=1)
        # outputs = self.model_dict(target_features)
        # _, target_cls = torch.max(outputs, 1)
        #####################
        # N = 5

        # def f(x):
        #     return x ** 2
        #
        # x = torch.randn(N, requires_grad=True)
        # y = f(x)
        # print(x)
        # print(y)

        # stList = []
        # stResultList=[]
        # for i in range(N):
        #     t = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
        #     t.requires_grad = True
        #     stList.append(t)
        #     ev=evaluate_modify(compute_traj(t), target_pos, class_scores[target_cls], RADIUS)
        #     stResultList.append(ev)
        #
        # x = torch.stack((stList[0], stList[1], stList[2], stList[3], stList[N - 1]), dim=0)
        # y=torch.stack((stResultList[0], stResultList[1], stResultList[2], stResultList[3], stResultList[N - 1]), dim=0)
        # print(x)
        # print(y)
        # # y=evaluate_modify(compute_traj(x), target_pos, class_scores[target_cls], RADIUS)
        # basis_vectors = torch.eye(N)
        # # jacobian_rows = [torch.autograd.grad(y, x, v, retain_graph=True,allow_unused=True)[0]
        # #                  for v in basis_vectors.unbind()]
        # # jacobian = torch.stack(jacobian_rows)
        #
        # def get_vjp(v):
        #     return torch.autograd.grad(y, x, v)[0]
        #
        # jacobian_vmap = vmap(get_vjp)(basis_vectors)
        #################
        ##迭代

        maxScore = float("-inf")
        result = None
        # start1
        ctps_inter = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
        ctps_inter.requires_grad = True
        # optimizer = optim.SGD(params=[ctps_inter], lr=0.0001, momentum=1, dampening=0, weight_decay=0.01, nesterov=True)
        # optimizer = optim.Adam(params=[ctps_inter], lr=0.6,maximize=True)

        latest_score_modify = None

        #
        lr = 0.65

        while time.time() - start < 0.2999:
            # start1
            score_modify = evaluate_modify(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
            # optimizer.zero_grad()
            score_modify.backward()
            # optimizer.step()
            ctps_inter.data = ctps_inter.data + lr * ctps_inter.grad / torch.norm(ctps_inter.grad)
            score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)

            if score > maxScore:
                maxScore = score
                result = ctps_inter.data

            if latest_score_modify == None:
                latest_score_modify = score_modify
            else:
                if abs(latest_score_modify - score_modify) <= 0.3:

                    ctps_inter = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
                    ctps_inter.requires_grad = True
                latest_score_modify = score_modify


        return result


def evaluate_modify(
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
        radius: float,
) -> torch.Tensor:
    """Evaluate the trajectory and return the score it gets.

    Parameters
    ----------
    traj: Tensor of shape `(*, T, 2)`
        The discretized trajectory, where `*` is some batch dimension and `T` is the discretized time dimension.
    target_pos: Tensor of shape `(N, 2)`
        x-y positions of shape where `N` is the number of targets.
    target_scores: Tensor of shape `(N,)`
        Scores you get when the corresponding targets get hit.
    """
    cdist = torch.cdist(target_pos, traj)  # see https://pytorch.org/docs/stable/generated/torch.cdist.html
    d = cdist.min(-1).values
    hit = (d <= radius)
    d[hit] = -target_scores[hit].float()
    d[~hit] = radius / d[~hit]

    value = torch.sum(d * target_scores, dim=-1)
    return value


P = 3  # spline degree
N_CTPS = 5  # number of control points

RADIUS = 0.3
N_CLASSES = 10
FEATURE_DIM = 256


def generate_game(
        n_targets: int,
        n_ctps: int,
        feature: torch.Tensor,
        label: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """
    Randomly generate a task configuration.
    """
    assert len(feature) == len(label)

    sample_indices = torch.randperm(len(feature))[:n_targets]
    target_pos = torch.rand((n_targets, 2)) * torch.tensor([n_ctps - 2, 2.]) + torch.tensor([1., -1.])
    target_features = feature[sample_indices]
    target_cls = label[sample_indices]
    class_scores = torch.randint(-N_CLASSES, N_CLASSES, (N_CLASSES,))

    return target_pos, target_features, target_cls, class_scores


def compute_traj(ctps_inter: torch.Tensor):
    """Compute the discretized trajectory given the second to the second control points"""
    t = torch.linspace(0, N_CTPS - P, 100, device=ctps_inter.device)
    knots = torch.cat([
        torch.zeros(P, device=ctps_inter.device),
        torch.arange(N_CTPS + 1 - P, device=ctps_inter.device),
        torch.full((P,), N_CTPS - P, device=ctps_inter.device),
    ])
    ctps = torch.cat([
        torch.tensor([[0., 0.]], device=ctps_inter.device),
        ctps_inter,
        torch.tensor([[N_CTPS, 0.]], device=ctps_inter.device)
    ])
    return splev(t, knots, ctps, P)


def evaluate(
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
        radius: float,
) -> torch.Tensor:
    """Evaluate the trajectory and return the score it gets.

    Parameters
    ----------
    traj: Tensor of shape `(*, T, 2)`
        The discretized trajectory, where `*` is some batch dimension and `T` is the discretized time dimension.
    target_pos: Tensor of shape `(N, 2)`
        x-y positions of shape where `N` is the number of targets.
    target_scores: Tensor of shape `(N,)`
        Scores you get when the corresponding targets get hit.
    """
    cdist = torch.cdist(target_pos, traj)  # see https://pytorch.org/docs/stable/generated/torch.cdist.html
    d = cdist.min(-1).values
    hit = (d < radius)
    value = torch.sum(hit * target_scores, dim=-1)
    return value


def splev(
        x: torch.Tensor,
        knots: torch.Tensor,
        ctps: torch.Tensor,
        degree: int,
        der: int = 0
) -> torch.Tensor:
    """Evaluate a B-spline or its derivatives.

    See https://en.wikipedia.org/wiki/B-spline for more about B-Splines.
    This is a PyTorch implementation of https://en.wikipedia.org/wiki/De_Boor%27s_algorithm

    Parameters
    ----------
    x : Tensor of shape `(t,)`
        An array of points at which to return the value of the smoothed
        spline or its derivatives.
    knots: Tensor of shape `(m,)`
        A B-Spline is a piece-wise polynomial.
        The values of x where the pieces of polynomial meet are known as knots.
    ctps: Tensor of shape `(n_ctps, dim)`
        Control points of the spline.
    degree: int
        Degree of the spline.
    der: int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k, the degree of the spline).
    """
    if der == 0:
        return _splev_torch_impl(x, knots, ctps, degree)
    else:
        assert der <= degree, "The order of derivative to compute must be less than or equal to k."
        n = ctps.size(-2)
        ctps = (ctps[..., 1:, :] - ctps[..., :-1, :]) / (knots[degree + 1:degree + n] - knots[1:n]).unsqueeze(-1)
        return degree * splev(x, knots[..., 1:-1], ctps, degree - 1, der - 1)


def _splev_torch_impl(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int):
    """
        x: (t,)
        t: (m, )
        c: (n_ctps, dim)
    """
    assert t.size(0) == c.size(0) + k + 1, f"{len(t)} != {len(c)} + {k} + {1}"  # m= n + k + 1

    x = torch.atleast_1d(x)
    assert x.dim() == 1 and t.dim() == 1 and c.dim() == 2, f"{x.shape}, {t.shape}, {c.shape}"
    n = c.size(0)
    u = (torch.searchsorted(t, x) - 1).clip(k, n - 1).unsqueeze(-1)
    x = x.unsqueeze(-1)
    d = c[u - k + torch.arange(k + 1, device=c.device)].contiguous()
    for r in range(1, k + 1):
        j = torch.arange(r - 1, k, device=c.device) + 1
        t0 = t[j + u - k]
        t1 = t[j + u + 1 - r]
        alpha = ((x - t0) / (t1 - t0)).unsqueeze(-1)
        d[:, j] = (1 - alpha) * d[:, j - 1] + alpha * d[:, j]
    return d[:, k]



