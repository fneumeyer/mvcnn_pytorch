from tools.the_dataset import TheDataset
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

from tools.Trainer import SpaPGNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN
from models.SpaPGNet import SpaPGNet

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=5)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=0)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
# parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-dataset_path2d", type=str, default="modelnet40_images_new_12x")
parser.add_argument("-dataset_path3d", type=str, default="ModelNet40Voxelized")
parser.set_defaults(train=False)

def create_folder(log_dir):
    
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    args = parser.parse_args()

    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    # STAGE 2
    log_dir = args.name+'_stage_2'
    create_folder(log_dir)
    # cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)
    n_models_train = args.num_models*args.num_views
    # cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    # del cnet
    spa_pg_net = SpaPGNet(args.name)

    optimizer = optim.Adam(spa_pg_net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    # TODO
    train_dataset = TheDataset(args.dataset_path2d, args.dataset_path3d, split='overfit', num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=0)# shuffle needs to be false! it's done within the trainer
    val_dataset = TheDataset(args.dataset_path2d, args.dataset_path3d, split='overfit', num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=True, num_workers=0)
    print('num_train_files: '+str(len(train_dataset)))
    print('num_val_files: '+str(len(val_dataset)))
    
    trainer = SpaPGNetTrainer(spa_pg_net, train_loader, val_loader, optimizer, nn.BCEWithLogitsLoss(), 'SpaPGNet', log_dir, num_views=args.num_views)
    trainer.train(300)
