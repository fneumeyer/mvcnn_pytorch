import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
import time

class SpaPGNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views

        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)


    def train(self, n_epochs):

        best_iou = 0
        i_accumulated = 0
        self.model.train()
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

        for epoch in range(n_epochs):
            # permute data for mvcnn
            # TODO this shuffling done here seems to make no sense, change it
            # rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            # filepaths_new = []
            # for i in range(len(rand_idx)):
            #     filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            # self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):
                # TODO check what the data tuple exactly contains
                # TODO right now it is (target_class, tensor of size [1, 12, 3, 224, 224], list of 12 filepaths)

                # in_data has shape B, L, C, H, W: Batch size, number of images, number of channels per image, height, width
                label, images, grid = data
                in_data = Variable(images).to(device)
                target = Variable(grid).to(device).float()

                self.optimizer.zero_grad()

                # assumption: out_data.shape = [B, 1, 64, 64, 64]
                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)
                
                self.writer.add_scalar('train/train_loss', loss, i_accumulated+i+1)

                prediction = torch.greater(out_data, 0).float()
                intersection = torch.sum(prediction.mul(target)).float()
                union = torch.sum(torch.ge(prediction.add(target), 1))
                iou = intersection / union;
                self.writer.add_scalar('train/train_iou', iou, i_accumulated+i+1)

                loss.backward()
                self.optimizer.step()
                
                log_str = 'epoch %d, step %d: train_loss %.3f; train_iou %.3f' % (epoch+1, i+1, loss, iou)
                if (i+1)%1==0:
                    print(log_str)
            i_accumulated += i

            # evaluation
            if (epoch+1)%1==0:
                with torch.no_grad():
                    loss, val_overall_iou, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
                self.writer.add_scalar('val/val_overall_iou', val_overall_iou, epoch+1)
                self.writer.add_scalar('val/val_loss', loss, epoch+1)

            # save best model
            if val_overall_iou > best_iou:
                best_iou = val_overall_iou
                self.model.save(self.log_dir, epoch)
 
            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir+"/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        # in_data = None
        # out_data = None
        # target = None
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')


        wrong_class = np.zeros(2)
        samples_class = np.zeros(2)
        all_loss = 0
        sum_iou = 0
        n_iou = 0
        n_samples = 0

        self.model.eval()

        for _, data in enumerate(self.val_loader, 0):

            # in_data has shape B, L, C, H, W: Batch size, number of images, number of channels per image, height, width

            label, images, grid = data
            in_data = Variable(images).to(device)
            target = Variable(grid).to(device).float()
            out_data = self.model(in_data)

            prediction = torch.greater(out_data, 0).float()
            intersection = torch.sum(prediction.mul(target)).float()
            union = torch.sum(torch.ge(prediction.add(target), 1)).float()
            iou = intersection.float() / union.float();
            sum_iou += iou
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            n_iou += 1
            n_samples += images.size()[0]

        print ('Total # of test models: ', n_samples)
        val_mean_class_acc = 0 #np.mean((samples_class-wrong_class)/samples_class)
        val_overall_iou = (sum_iou / n_iou).cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall iou. : ', val_overall_iou)
        print ('val loss : ', loss)

        self.model.train()

        return loss, val_overall_iou, val_mean_class_acc

