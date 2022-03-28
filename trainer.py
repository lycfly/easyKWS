"""
model trainer
"""
from torch.autograd import Variable
from data import get_label_dict, get_wav_list, SpeechDataset, get_semi_list
from torch.utils.data import DataLoader
import torch
from time import time
from torch.nn import Softmax
import numpy as np
from dataset_prepare import *
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import math
from tqdm import *
from confusion_matrix import confusion_matrix_generate, FAFR_rating
from random import choice
from send_message import send_message
from nets import fix_bn_my

def get_model(model=None, m=False, pretrained=False, label_num=12, finetune=False):
    mdl = torch.nn.DataParallel(model(label_num=label_num)) if m else model(label_num=label_num, finetune=finetune)
    if not pretrained:
        return mdl
    else:
        print("load pretrained model here...")
        mdl.load_state_dict(torch.load(pretrained))
        return mdl

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def train_one_epoch(global_step, epoch, iters, speechmodel, train_dataloader,
                    loss_fn, optimizer, warmup, lr_scheduler, writer):
    total_correct = 0
    num_labels = 0
    it = 0
    running_loss = 0.0
    speechmodel.train()
    pbar = tqdm(train_dataloader,unit="audios", unit_scale=train_dataloader.batch_size, ncols=150)

    for i, batch_data in enumerate(pbar):
        spec = batch_data['spec']
        label = batch_data['label']
        spec, label = Variable(spec.cuda()), Variable(label.cuda())
        y_pred = speechmodel(spec)
        loss = loss_fn(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if warmup:
            lr_scheduler.step(epoch+i/iters)

        '''Average acc and loss'''
        it += 1
        global_step += 1
        running_loss += loss.item()

        _, pred_labels = torch.max(y_pred.data, 1)
        correct = (pred_labels == label.data).sum()
        total_correct += correct
        num_labels += len(label)
        acc = (100. * total_correct / num_labels)

        pbar.set_postfix({
            'Train loss' : "%.05f" % (running_loss / it),
            'Train acc' : "%.02f" % (acc.cpu().numpy()),
            'lr' : "%.06f" % (get_lr(optimizer)),
        })
        writer.add_scalar('%s/learning_rate' % 'train', get_lr(optimizer), global_step)
        writer.add_scalar('%s/loss' % 'train', loss.item(), global_step)
    writer.add_scalar('%s/epoch_accuracy' % 'train', acc.cpu().numpy(), epoch)
    writer.add_scalar('%s/epoch_loss' % 'train', running_loss / it, epoch)
    return global_step

def valid(epoch, speechmodel, val_dataloader, loss_fn, words, CODER, best_accuracy, best_loss, model_path, writer):
    total_correct = 0
    num_labels = 0
    it = 0
    running_loss = 0.0
    speechmodel.eval()
    speechmodel.step = epoch
    pbar = tqdm(val_dataloader, unit="audios", unit_scale=val_dataloader.batch_size, ncols=120)
    cm_lp = np.zeros([len(words) + 2, len(words) + 2]).astype(np.int32)
    for bat, batch_data in enumerate(pbar):
        '''
        Forward and backward propagate
        '''
        if bat == len(pbar)-1:
            speechmodel.ifwriter = True
        spec = batch_data['spec']
        label = batch_data['label']
        spec, label = Variable(spec.cuda()), Variable(label.cuda())
        y_pred = speechmodel(spec)
        _, pred_labels = torch.max(y_pred.data, 1)
        correct = (pred_labels == label.data).sum()
        loss = loss_fn(y_pred, label)
        '''Average acc and loss'''
        it += 1
        running_loss += loss.item()
        total_correct += correct
        num_labels += len(label)
        acc = (100. * total_correct / num_labels)

        cm_lp += confusion_matrix_generate(label.cpu().data.numpy(), pred_labels.cpu().data.numpy(),np.arange(len(words)+2))
        FAR, FRR = FAFR_rating(cm_lp)
        pbar.set_postfix({
            'Val loss' : "%.05f" % (running_loss / it),
            'Val acc'  : "%.02f" % (acc.cpu().numpy()),
            'FAR'      : "%.05f" % (100.*FAR),
            'FRR'      : "%.05f" % (100.*FRR),
        })
        speechmodel.ifwriter = False
    accuracy = total_correct / num_labels
    epoch_loss = running_loss / it
    writer.add_scalar('%s/epoch_accuracy' % 'val', acc.cpu().numpy(), epoch)
    writer.add_scalar('%s/epoch_loss' % 'val', epoch_loss, epoch)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(speechmodel.state_dict(), model_path)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
    print(cm_lp)
    return best_accuracy, best_loss, epoch_loss

def final_test(speechmodel, test_dataloader, loss_fn, words):
    total_correct = 0
    num_labels = 0
    it = 0
    running_loss = 0.0
    speechmodel.eval()
    #speechmodel.step = epoch
    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size, ncols=120)
    cm_lp = np.zeros([len(words) + 2, len(words) + 2]).astype(np.int32)
    for bat, batch_data in enumerate(pbar):
        spec = batch_data['spec']
        label = batch_data['label']
        spec, label = Variable(spec.cuda()), Variable(label.cuda())
        y_pred = speechmodel(spec)
        _, pred_labels = torch.max(y_pred.data, 1)
        correct = (pred_labels == label.data).sum()
        loss = loss_fn(y_pred, label)
        '''Average acc and loss'''
        it += 1
        running_loss += loss.item()
        total_correct += correct
        num_labels += len(label)
        acc = (100. * total_correct / num_labels)
        cm_lp += confusion_matrix_generate(label.cpu().data.numpy(), pred_labels.cpu().data.numpy(),np.arange(len(words)+2))
        FAR, FRR = FAFR_rating(cm_lp)
        pbar.set_postfix({
            'Val loss' : "%.05f" % (running_loss / it),
            'Val acc'  : "%.02f" % (acc.cpu().numpy()),
            'FAR'      : "%.05f" % (100.*FAR),
            'FRR'      : "%.05f" % (100.*FRR),
        })
        speechmodel.ifwriter = False
    accuracy = total_correct / num_labels
    epoch_loss = running_loss / it
    print(cm_lp)
    return accuracy, epoch_loss, cm_lp


def train_model(log_path, model_path, data_path, noise_path, words, num_workers, target_dataset, model_class, unknown_ratio, preprocess_fun, is_1d, reshape_size, BATCH_SIZE, epochs, CODER, learning_rate=0.01,
               feature_cfg={},bagging_num=1, pretrained=None, pretraining=False, MGPU=False, warmup=False, **args):
    """
    :param model_class: model class. e.g. vgg, resnet, senet
    :param preprocess_fun: preprocess function. e.g. mel, mfcc, raw wave
    :param is_1d: boolean. True for conv1d models and false for conv2d
    :param reshape_size: int. only for conv2d, reshape the image size
    :param BATCH_SIZE: batch size.
    :param epochs: number of epochs
    :param CODER: string for saving and loading model/files
    :param preprocess_param: parameters for preprocessing function
    :param bagging_num: number of training per model, aka bagging models
    :param pretrained: path to pretrained model
    :param pretraining: boolean. if this is pretraining
    :param MGPU: whether using multiple gpus
    """

    full_name = '%s_bs%s_lr%.1e' % (
        CODER, BATCH_SIZE, learning_rate)
    writer = SummaryWriter(os.path.join(log_path,full_name), comment=('_speech_commands_' + full_name))

    label_to_int, int_to_label = get_label_dict(words)

    train_list, val_list, test_list = get_wav_list(path=data_path, target_dataset=target_dataset, words=label_to_int.keys(),
                                                  unknown_ratio=unknown_ratio)

    traindataset = SpeechDataset(noise_path=noise_path, mode='train', label_words_dict=label_to_int,
                                 wav_list=train_list, add_noise=True, preprocess_fun=preprocess_fun,
                                 feature_cfg=feature_cfg, resize_shape=reshape_size,is_1d=is_1d)
 
    valdataset   = SpeechDataset(noise_path=noise_path, mode='val', label_words_dict=label_to_int,
                                 wav_list=val_list, add_noise=False, preprocess_fun=preprocess_fun,
                                 feature_cfg=feature_cfg, resize_shape=reshape_size,is_1d=is_1d)
   
    testdataset  = SpeechDataset(noise_path=noise_path, mode='test', label_words_dict=label_to_int,
                                 wav_list=test_list, add_noise=False, preprocess_fun=preprocess_fun,
                                 feature_cfg=feature_cfg, resize_shape=reshape_size,is_1d=is_1d)
    
    trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True, num_workers = num_workers)
    valloader   = DataLoader(valdataset,   BATCH_SIZE, shuffle=False, num_workers = num_workers)
    testloader  = DataLoader(testdataset,  BATCH_SIZE, shuffle=False, num_workers = num_workers)

    iters = len(trainloader)

    loss_fn = torch.nn.CrossEntropyLoss()
    model_path_full = os.path.join(model_path,"model_best_acc_%s.pth" %(CODER))
    speechmodel = get_model(model=model_class, m=MGPU, pretrained=pretrained, label_num=len(words)+2)
    speechmodel = speechmodel.cuda()
    speechmodel.writer = writer

    if warmup:
        T_max = epochs
        warmup_iter = 5
        lr_max = 0.1
        lr_min = 1e-5
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speechmodel.parameters()),
                                    lr=learning_rate, momentum=0.9,weight_decay=0.001)
        lambda0 = lambda cur_iter:cur_iter/warmup_iter if cur_iter<warmup_iter \
            else (lr_min+0.5*(lr_max-lr_min)*(1.0 + math.cos((cur_iter-warmup_iter)/(T_max-warmup_iter)*math.pi)))/0.1
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speechmodel.parameters()),
                                    lr=learning_rate, momentum=0.9, weight_decay=0.00001)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones[20,70], gamma=0.1)

    since = time()
    global_step = 0
    best_accuracy = 0
    best_loss = float("inf")
    training = 1
    if training == 1:
        for e in range(epochs):
            print("epoch %3d with lr=%.03e" % (e, get_lr(optimizer)))
            speechmodel.ifwriter = False
            global_step = train_one_epoch(global_step, e, iters, speechmodel, trainloader,loss_fn, optimizer, warmup, lr_scheduler, writer)
            best_accuracy, best_loss, epoch_loss = valid(e, speechmodel, valloader, loss_fn, words, CODER, best_accuracy, best_loss, model_path_full, writer)

            time_elapsed = time() - since
            timer_str = 'total time elasped: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed //3600, (time_elapsed % 3600)/60, time_elapsed % 60)
            print("%s, best accuracy: %.02f, best loss %f" % (timer_str,100*best_accuracy, best_loss))
            if not warmup:
                lr_scheduler.step()
            if target_dataset in ['ComVoice']: # update common voice unknown words
                train_list, _,_ = get_wav_list(path=data_path, target_dataset=target_dataset, words=label_to_int.keys(),unknown_ratio=unknown_ratio)
                traindataset = SpeechDataset(noise_path=noise_path, mode='train', label_words_dict=label_to_int, wav_list=train_list,
                                             add_noise=True, preprocess_fun=preprocess_fun, feature_cfg=feature_cfg,resize_shape=reshape_size,is_1d=is_1d)
                trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True, num_workers=num_workers)
    
    test_acc, test_loss, cm_lp = final_test(speechmodel, testloader, loss_fn, words)
    
    message = "Test accuracy: %.03f, Test loss %f" % (100 * test_acc, test_loss)
    print(message)
    message += "\\n" + str(cm_lp)
    send_message(message)
    writer.close()

 
def test_inference(log_path, model_path, data_path, noise_path, target_dataset, model_class, words, num_workers, unknown_ratio, preprocess_fun, is_1d, reshape_size, BATCH_SIZE, epochs, CODER, learning_rate=0.01,
               feature_cfg={},bagging_num=1, pretrained=None, pretraining=False, MGPU=False, warmup=False, **args):
    print("doing prediction...")
    label_to_int, int_to_label = get_label_dict(words)

    _, _, test_list = get_wav_list(path=data_path, target_dataset=target_dataset, words=label_to_int.keys(),
                                                  unknown_ratio=unknown_ratio)
    testdataset  = SpeechDataset(noise_path=noise_path, mode='test', label_words_dict=label_to_int,
                                 wav_list=test_list, add_noise=False, preprocess_fun=preprocess_fun,
                                 feature_cfg=feature_cfg, resize_shape=reshape_size,is_1d=is_1d)
    testloader  = DataLoader(testdataset,  BATCH_SIZE, shuffle=False, num_workers = num_workers)

    loss_fn = torch.nn.CrossEntropyLoss()

    speechmodel = get_model(model=model_class, m=MGPU, pretrained=pretrained, label_num=len(words)+2)
    speechmodel.load_state_dict(torch.load(os.path.join(model_path,"model_best_acc_%s.pth" %(CODER))))
    speechmodel = speechmodel.cuda()
    speechmodel.eval()

    test_acc, test_loss, cm_lp = final_test(speechmodel, testloader, loss_fn, words)
    print("Test accuracy: %.03f, Test loss %f" % (100 * test_acc, test_loss))



def finetune(log_path, model_path, data_path, noise_path, words, num_workers, target_dataset, model_class, unknown_ratio, preprocess_fun, is_1d, reshape_size, BATCH_SIZE, epochs_ft, CODER, learning_rate_ft=0.0001,
               feature_cfg={},bagging_num=1, pretrained=None, pretraining=False, MGPU=False, warmup=False, **args):
    """
    :param model_class: model class. e.g. vgg, resnet, senet
    :param preprocess_fun: preprocess function. e.g. mel, mfcc, raw wave
    :param is_1d: boolean. True for conv1d models and false for conv2d
    :param reshape_size: int. only for conv2d, reshape the image size
    :param BATCH_SIZE: batch size.
    :param epochs_ft: number of epochs_ft
    :param CODER: string for saving and loading model/files
    :param preprocess_param: parameters for preprocessing function
    :param bagging_num: number of training per model, aka bagging models
    :param pretrained: path to pretrained model
    :param pretraining: boolean. if this is pretraining
    :param MGPU: whether using multiple gpus
    """

    full_name = 'finetune_%s_bs%s_lr%.1e' % (
        CODER, BATCH_SIZE, learning_rate_ft)
    writer = SummaryWriter(os.path.join(log_path,full_name), comment=('_speech_commands_' + full_name))

    label_to_int, int_to_label = get_label_dict(words)

    train_list, val_list, test_list = get_wav_list(path=data_path, target_dataset=target_dataset, words=label_to_int.keys(),
                                                  unknown_ratio=unknown_ratio)

    traindataset = SpeechDataset(noise_path=noise_path, mode='train', label_words_dict=label_to_int,
                                 wav_list=train_list, add_noise=True, preprocess_fun=preprocess_fun,
                                 feature_cfg=feature_cfg, resize_shape=reshape_size,is_1d=is_1d)
 
    valdataset   = SpeechDataset(noise_path=noise_path, mode='val', label_words_dict=label_to_int,
                                 wav_list=val_list, add_noise=False, preprocess_fun=preprocess_fun,
                                 feature_cfg=feature_cfg, resize_shape=reshape_size,is_1d=is_1d)
   
    testdataset  = SpeechDataset(noise_path=noise_path, mode='test', label_words_dict=label_to_int,
                                 wav_list=test_list, add_noise=False, preprocess_fun=preprocess_fun,
                                 feature_cfg=feature_cfg, resize_shape=reshape_size,is_1d=is_1d)
    
    trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True, num_workers = num_workers)
    valloader   = DataLoader(valdataset,   BATCH_SIZE, shuffle=False, num_workers = num_workers)
    testloader  = DataLoader(testdataset,  BATCH_SIZE, shuffle=False, num_workers = num_workers)

    iters = len(trainloader)

    loss_fn = torch.nn.CrossEntropyLoss()
    pretrained_path = os.path.join(model_path,"model_best_acc_%s.pth" %(CODER))
    finetune_path = os.path.join(model_path,"model_best_acc_finetune_%s.pth" %(CODER))
    speechmodel = get_model(model=model_class, m=MGPU, pretrained=pretrained_path, label_num=len(words)+2, finetune = True)
    speechmodel = speechmodel.cuda()
    speechmodel.train()
    speechmodel.apply(fix_bn_my) # fix batchnorm
    speechmodel.writer = writer

    
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speechmodel.parameters()),
                                lr=learning_rate_ft, momentum=0.9, weight_decay=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1, last_epoch=-1)

    since = time()
    global_step = 0
    best_accuracy = 0
    best_loss = float("inf")
    training = 1
    if training == 1:
        for e in range(epochs_ft):
            print("epoch %3d with lr=%.03e" % (e, get_lr(optimizer)))
            speechmodel.ifwriter = False
            global_step = train_one_epoch(global_step, e, iters, speechmodel, trainloader,loss_fn, optimizer, False, lr_scheduler, writer)
            best_accuracy, best_loss, epoch_loss = valid(e, speechmodel, valloader, loss_fn, words, CODER, best_accuracy, best_loss, finetune_path, writer) #finetune path

            time_elapsed = time() - since
            timer_str = 'total time elasped: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed //3600, (time_elapsed % 3600)/60, time_elapsed % 60)
            print("%s, best accuracy: %.02f, best loss %f" % (timer_str,100*best_accuracy, best_loss))
            lr_scheduler.step()

            if target_dataset in ['ComVoice']: # update common voice unknown words
                train_list, _,_ = get_wav_list(path=data_path, target_dataset=target_dataset, words=label_to_int.keys(),unknown_ratio=unknown_ratio)
                traindataset = SpeechDataset(noise_path=noise_path, mode='train', label_words_dict=label_to_int, wav_list=train_list,
                                             add_noise=True, preprocess_fun=preprocess_fun, feature_cfg=feature_cfg,resize_shape=reshape_size,is_1d=is_1d)
                trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True, num_workers=num_workers)
    
    test_acc, test_loss, cm_lp = final_test(speechmodel, testloader, loss_fn, words)

    message = "Finetune Test accuracy: %.03f, Test loss %f\n" % (100 * test_acc, test_loss)
    print(message)
    message += "\\n" + str(cm_lp)
    send_message(message)
    writer.close()