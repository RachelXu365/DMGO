# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

# utils
import os
import datetime
import numpy as np
import joblib

from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake
from model.FusAtNet import FusAtNet
from model.EndNet import EndNet
from model.DML_Hong import Early_fusion_CNN, Middle_fusion_CNN, Late_fusion_CNN, Cross_fusion_CNN
from model.S2ENet import S2ENet
from model.OGMAblationS2ENet import OGMAblationS2ENet
from model.dmgo import DMGO
from model.MUNet import MUNet
from losses import Cross_fusion_CNN_Loss, EndNet_Loss


def get_model(name, **kwargs):
    device = kwargs.setdefault("device", torch.device("cuda"))
    n_classes = kwargs["n_classes"]
    (n_bands, n_bands2) = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    if name == "FusAtNet":
        kwargs.setdefault("patch_size", 11)
        center_pixel = True
        model = FusAtNet(n_bands, n_bands2, n_classes)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 150)
        kwargs.setdefault("batch_size", 64)
    elif name == "S2ENet":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        model = S2ENet(n_bands, n_bands2, n_classes, kwargs["patch_size"])
        if(kwargs['modulation']=='Normal'):
            lr = kwargs.setdefault("lr", 0.001)
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            lr = kwargs.setdefault("lr", 0.001)
            lr_decay_step = kwargs["lr_decay_step"]
            lr_decay_ratio = kwargs["lr_decay_ratio"]
            optimizer = optim.Adam(model.parameters(), lr=lr)
          #  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
          #  kwargs.setdefault("scheduler", optim.lr_scheduler.StepLR(optimizer, lr_decay_step, lr_decay_ratio))
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
    elif name == 'DMGO':
        center_pixel = True
        model = DMGO(n_bands, n_bands2, n_classes, kwargs["patch_size"])
        optimizer = optim.Adam(model.parameters(), lr=kwargs["lr"])
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])

    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    lr_decay_step = kwargs["lr_decay_step"]
    lr_decay_ratio = kwargs["lr_decay_ratio"]
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs["epoch"]),
    )
    kwargs.setdefault('scheduler', None)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs



def train(
    args,
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cpu"),
    display=None,
    val_loader=None,
):
    train_loss_list = []


    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    save_epoch = epoch // 20 if epoch > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []
    best_acc = 0.0
    flag = 0
    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        modal1 = []
        modal2 = []
        softmax = nn.Softmax(dim=1)
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()

        net.train()
        avg_loss = 0.0

        _loss = 0
        _loss_seam = 0
        _loss_seem = 0

        for batch_idx, (data, data2, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            data, data2, target = data.to(device), data2.to(device), target.to(device)

            optimizer.zero_grad()
            
            output, out_ogm, logp_intr, p_hsi, p_lidar = net(data, data2)
            if args['modulation'] == 'OGM' or args['modulation'] == 'Normal':
                weight_size = net.fc.weight.size(1)
                hsi = (torch.mm(out_ogm[:,:weight_size//2], torch.transpose(net.fc.weight[:,:weight_size//2], 0, 1)) + net.fc.bias/2)
                lidar = (torch.mm(out_ogm[:,weight_size//2:], torch.transpose(net.fc.weight[:,weight_size//2:], 0, 1)) + net.fc.bias/2)
                score_hsi = sum([softmax(hsi)[i][target[i]] for i in range(hsi.size(0))])
                score_lidar = sum([softmax(lidar)[i][target[i]] for i in range(lidar.size(0))])
                ratio_seam = score_hsi / score_lidar
                ratio_seem = 1/ratio_seam
                beta = tanh(score_hsi / (score_hsi + score_lidar))
                
                if ratio_seam > 1:
                    coeff_seam = 1 - tanh(args['alpha'] * relu(ratio_seam))
                    coeff_seem = 1
                    modal1.append(coeff_seam)

                else:
                    coeff_seem = 1 - tanh(args['alpha'] * relu(ratio_seem))
                    coeff_seam = 1
                    modal2.append(coeff_seem)

            loss_BCE = criterion(output, target)
            loss_AVM = 0.1*F.kl_div(logp_intr, p_hsi, reduction='mean')
            loss = loss_BCE + loss_AVM
            loss.backward()
            if args['modulation']=='OGM':
                # Modulation starts here !
                if args['modulation_starts'] <= e <= args['modulation_ends']:
                    for name, parms in net.named_parameters():
                        layer = str(name).split('.')[0]
                        # print(name)
                        if '_a' in layer and len(parms.grad.size()) >= 4:
                            if args['modulation'] == 'OGM':
                                parms.grad *= coeff_seam

                        if '_b' in layer and len(parms.grad.size()) >= 4:
                            if args['modulation'] == 'OGM':
                                parms.grad *= coeff_seem
                else:
                    pass

            optimizer.step()

            _loss += loss.item()
            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter : iter_],
                    win=loss_win,
                    update=update,
                    opts={
                        "title": "Training loss",
                        "xlabel": "Iterations",
                        "ylabel": "Loss",
                    },
                )
                tqdm.write(string)

                train_loss_list.append(avg_loss)

                if len(val_accuracies) > 0:
                    val_win = display.line(
                        Y=np.array(val_accuracies),
                        X=np.arange(len(val_accuracies)),
                        win=val_win,
                        opts={
                            "title": "Validation accuracy",
                            "xlabel": "Epochs",
                            "ylabel": "Accuracy",
                        },
                    )
            iter_ += 1
            del (data, target, loss, output)


        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
            string = "valid \taccuracies: {:.6f}"
            string = string.format(val_acc)
            tqdm.write(string)
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if val_acc > best_acc and e > 80:
            best_acc = val_acc
            print("best_acc is %.8f, and save model successfully" % (best_acc))
            save_model(
                net,
                camel_to_snake(str(net.__class__.__name__)),
                data_loader.dataset.name,
                epoch=e,
                metric=abs(metric))


def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
    """
    Using strftime in case it triggers exceptions on windows 10 system
    """
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = time_str + "_epoch{epoch}_{metric:.4f}".format(
            **kwargs
        )
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        filename = time_str
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + ".pkl")


def val(net, data_loader, device="cpu", supervision="full"):
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    net.eval()
    for batch_idx, (data, data2, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, data2, target = data.to(device), data2.to(device), target.to(device)
            if supervision == "full":
                output = net(data, data2)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs

            if isinstance(output, tuple):   # For multiple outputs
                output = output[0]
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total
