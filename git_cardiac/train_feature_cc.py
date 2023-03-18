import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import SliceData,KneeData
from models import DCTeacherNet,DCStudentNet
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from functools import reduce

from utils import CriterionPairWiseforWholeFeatAfterPool, CriterionPairWiseforWholeFeatAfterPoolFeatureMaps  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###################################################** DATA LOADER **###############################################

def create_datasets(args):
    
    acc_factors = args.acceleration_factor.split(',')
    mask_types = args.mask_type.split(',')
    dataset_types = args.dataset_type.split(',')

    train_data = SliceData(args.train_path,acc_factors, dataset_types,mask_types,'train', args.usmask_path)
    dev_data = SliceData(args.validation_path,acc_factors,dataset_types,mask_types,'validation', args.usmask_path)

    return dev_data, train_data

def create_data_loaders(args):

    if args.dataset_type == 'knee':
        dev_data, train_data = create_datasets_knee(args)
    else:
        dev_data, train_data = create_datasets(args)

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=64,
        #pin_memory=True,
    )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        #num_workers=64,
        #pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        #num_workers=64,
        #pin_memory=True,
    )
    return train_loader, dev_loader, display_loader #,train_loader_error

###################################################** DISTILLATION **###############################################

#     @staticmethod
def compute_loss(s, t):
    return (s - t).pow(2).mean()

#     @staticmethod
def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

def correlaion_congruence(f_s, f_t):
#     f_s = f_s[:,:,5:-5,5:-5]
#     f_t = f_t[:,:,5:-5,5:-5]
    delta = torch.abs(f_s - f_t)
    #print(f'delta_shape is {delta.shape}')
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss

###################################################** TRAIN EPOCH **###############################################

def train_epoch(args, epoch,modelT,modelS,data_loader, optimizer, writer):#,error_range):# , vgg):
    
    modelT.eval() 
    modelS.train()

    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    alpha = 0.5
    ssim_factor = 0.1 

    loop = tqdm(data_loader)                   
    for iter, data in enumerate(loop):

        input,input_kspace,target,mask = data  
        input = input.unsqueeze(1).to(args.device)
        input_kspace = input_kspace.unsqueeze(1).to(args.device)
        target = target.unsqueeze(1).to(args.device)
        mask = mask.unsqueeze(1).to(args.device)

        input = input.float()
        target = target.float()
        mask = mask.float()

        outputT = modelT(input,input_kspace,mask)
        outputS = modelS(input,input_kspace,mask)

        outputT_feat = [outputT[0][-1],outputT[1][-1],outputT[2][-1],outputT[3][-1],outputT[4][-1]]
        outputS_feat = [outputS[0][-1],outputS[1][-1],outputS[2][-1],outputS[3][-1],outputS[4][-1]]

        loss_t = [correlaion_congruence(s,t) for s, t in zip(outputT_feat,outputS_feat)]
        loss   = reduce((lambda x,y : x + y),loss_t)  / len(loss_t)

        optimizer.zero_grad()
 
        loss.backward()

        optimizer.step()
        

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
#         break

    return avg_loss, time.perf_counter() - start_epoch

###################################################** EVALUATION **###############################################

def evaluate(args, epoch, modelT,modelS,data_loader, writer):

    modelT.eval()
    modelS.eval()

    losses_mse   = []
    losses_ssim  = []
 
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
    
            input,input_kspace, target,mask = data 
            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)
    
            input = input.float()
            target = target.float()
            mask = mask.float()

            outputT = modelT(input,input_kspace,mask)
            outputS = modelS(input,input_kspace,mask)

            outputT_feat = [outputT[0][-1],outputT[1][-1],outputT[2][-1],outputT[3][-1],outputT[4][-1]]
            outputS_feat = [outputS[0][-1],outputS[1][-1],outputS[2][-1],outputS[3][-1],outputS[4][-1]]
            

            loss_t = [correlaion_congruence(s,t) for s, t in zip(outputT_feat,outputS_feat)]
            loss   = reduce((lambda x,y : x + y),loss_t)  / len(loss_t)    
            losses_mse.append(loss.item())

        writer.add_scalar('Dev_Loss_mse',np.mean(losses_mse),epoch)
       
    return np.mean(losses_mse), time.perf_counter() - start

###################################################** VISUALIZE **###############################################

def visualize(args, epoch, model,data_loader, writer):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            input,input_kspace,target,mask = data 
            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)

            output = model(input.float(),input_kspace,mask.float())[-1]

            save_image(input, 'Input')
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target.float() - output.float()), 'Reconstruction error')

            #break

###################################################** SAVE MODEL **###############################################
            
def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
 
    modelT = DCTeacherNet(args).to(args.device)
    modelS = DCStudentNet(args).to(args.device)

    return modelT,modelS


def load_model(model,checkpoint_file):
  

    checkpoint = torch.load(checkpoint_file)
    model_wts = model.state_dict()
    for (tw,mw) in zip(checkpoint['model'],model_wts):
        if tw == mw:
            model_wts[mw] = checkpoint['model'][tw]

    model.load_state_dict(model_wts)

    return model


def get_error_range_teacher(loader,model):

    losses = []
    print ("Finding max and min error between teacher and target")

    for data in tqdm(loader):

        input,input_kspace, target,mask = data 
        input  = input.unsqueeze(1).float().to(args.device)
        input_kspace = input_kspace.unsqueeze(1).to(args.device)
        target = target.unsqueeze(1).float().to(args.device)
        mask = mask.unsqueeze(1).float().to(args.device)

        output = model(input,input_kspace,mask)[-1]
        
        loss = F.l1_loss(output,target).detach().cpu().numpy()

        losses.append(loss)

    min_error,max_error = np.min(losses),np.max(losses)
    
    return max_error 



def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer

###################################################** MAIN **###############################################
def main(args):

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    modelT,modelS = build_model(args)

    modelT = load_model(modelT,args.teacher_checkpoint)
    modelS = load_model(modelS,args.student_checkpoint)

    optimizer = build_optim(args, modelS.parameters())

    best_dev_loss = 1e9
    start_epoch = 0
    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
    for epoch in range(start_epoch, args.num_epochs):

        train_loss,train_time = train_epoch(args, epoch, modelT,modelS,train_loader,optimizer,writer)
        dev_loss,dev_time = evaluate(args, epoch, modelT, modelS, dev_loader, writer)
        scheduler.step()

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, modelS, optimizer,best_dev_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--batch-size', default=4, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=500, help='Period of loss reporting')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='us mask path')
    parser.add_argument('--teacher_checkpoint',type=str,help='teacher checkpoint')
    parser.add_argument('--mask_type',type=str,help='us mask path')
    parser.add_argument('--student_checkpoint',type=str,help='student checkpoint')
    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
