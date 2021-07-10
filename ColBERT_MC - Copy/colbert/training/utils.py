import os
import torch

from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)

def print_progress_MC(scores):
    positive_avgt, negative_avgt = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    positive_avg1, negative_avg1 = round(scores[:, 2].mean().item(), 2), round(scores[:, 3].mean().item(), 2)
    positive_avg2, negative_avg2 = round(scores[:, 4].mean().item(), 2), round(scores[:, 5].mean().item(), 2)
    positive_avg3, negative_avg3 = round(scores[:, 6].mean().item(), 2), round(scores[:, 7].mean().item(), 2)
    positive_avg4, negative_avg4 = round(scores[:, 8].mean().item(), 2), round(scores[:, 9].mean().item(), 2)
    print("#>>>   ", positive_avgt, negative_avgt, '\t\t|\t\t', positive_avgt - negative_avgt)
    print("#>>>   ", positive_avg1, negative_avg1, '\t\t|\t\t', positive_avg1 - negative_avg1)
    print("#>>>   ", positive_avg2, negative_avg2, '\t\t|\t\t', positive_avg2 - negative_avg2)
    print("#>>>   ", positive_avg3, negative_avg3, '\t\t|\t\t', positive_avg3 - negative_avg3)
    print("#>>>   ", positive_avg4, negative_avg4, '\t\t|\t\t', positive_avg4 - negative_avg4)


def manage_checkpoints(args, colbert, optimizer, batch_idx, Done = False):
    arguments = args.input_arguments.__dict__

    path = os.path.join(Run.path, 'checkpoints')

    if not os.path.exists(path):
        os.mkdir(path)

    if batch_idx % 2000 == 0:
        name = os.path.join(path, "colbert.dnn")
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)

    if batch_idx in SAVED_CHECKPOINTS:
        name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)

    if Done == True:
        name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)

def chosen_option(scores):
    index = [x.argmax().item() for x in scores]
    option = []
    for ind in index:
        if ind % 2 == 1:
            option.append(0)
        else:
            option.append(ind/2+1)
    return option

def chosen_option_test(scores):
    index = [x.argmax().item() for x in scores]
    option = []
    for ind in index:
        option.append(ind+1)
    return option
