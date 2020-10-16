import numpy as np
import torch

def add_l1_grads(l1_coef, param_groups):
    for group in param_groups:
        for p in group['params']:
            assert p.grad is not None, 'We have not decided yet what to do in this case'
            grad = p.grad.data
            grad.add_(l1_coef, torch.sign(p.data))


def validate_target_outupt_shapes(output, target):
    if output.ndimension() == 1:
        assert output.shape == target.shape
    else:
        assert output.ndimension() == 2
        assert output.shape[0] == target.shape[0]
        assert output.shape[1] > target.max()


def count_correct(output, target):
    if output.ndimension() == 1:
        pred = output >= 0.0
        correct_preds = pred.eq(target.view_as(pred))
        correct_batch = correct_preds.sum().item()
    elif output.ndimension() == 2 and target.ndimension() == 1:
        pred = output.argmax(dim=1, keepdim=True)
        correct_batch = pred.eq(target.view_as(pred)).sum().item()
    elif output.ndimension() == 2 and target.ndimension() == 2:
        pred = output.argmax(dim=1, keepdim=True)
        target_classes = target.argmax(dim=1, keepdim=True)
        correct_batch = pred.eq(target_classes.view_as(pred)).sum().item()
    return correct_batch

