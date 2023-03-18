import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_1 import SliceDataDev
from trial1 import DCTeacherNet,DCStudentNet#,IKD
import h5py
from tqdm import tqdm

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    print(reconstructions.keys())
    c=reconstructions['1.h5'].shape
    print(f'reconstructions shape is: {c}')

    for fname, recons in reconstructions.items():        
        with h5py.File(out_dir / fname, 'w') as f:
#             recons = recons.numpy()
#             print(f'recons shape is: {recons.shape}')
#             print(f'recons[45] shape is: {recons[45].shape}')
#             print(f'recons[45][1] shape is: {recons[45][1].shape}')
#             print(f'recons[45][0] shape is: {recons[45][0].shape}')

            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    #data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type,args.usmask_path)
    data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type,args.mask_type,args.usmask_path)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader

def create_data_loaders(args):

    if args.dataset_type == 'knee':
        data = KneeDataDev(args.data_path,args.acceleration_factor,args.dataset_type)
        data_loader = DataLoader(
            dataset=data,
            batch_size=args.batch_size,
            num_workers=1,
            
            pin_memory=True,)
    else:
        data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type,args.mask_type,args.usmask_path)
        data_loader = DataLoader(
            dataset=data,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=True,)

    return data_loader


def load_model(model_type,checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
#     print(args)

    if model_type == 'teacher':
        print('teacher')
        model = DCTeacherNet().to(args.device)
        model.load_state_dict(checkpoint['model'])
    elif model_type == 'student':
        print('student')
        model = DCStudentNet(args).to(args.device)
        model.load_state_dict(checkpoint['model'])
        
    elif model_type == 'teacherSFTN':
        model = DCTeacherNet(args).to(args.device)
        model_wts = model.state_dict()
        for mw in model_wts:
            for tw in checkpoint['model']:
                if tw == mw:
                    print(f'Loading weight {tw} to {mw}')
                    model_wts[mw] = checkpoint['model'][tw]
        model.load_state_dict(model_wts)

    elif model_type == 'studentSFTN':
        model = DCStudentNet(args).to(args.device)
        model_wts = model.state_dict()
        for mw in model_wts:
            for tw in checkpoint['model']:
                if tw == mw:
                    print(f'Loading weight {tw} to {mw}')
                    model_wts[mw] = checkpoint['model'][tw]
        model.load_state_dict(model_wts)
        
#     elif model_type == 'studentikd':
#         model = IKD().to(args.device)
# #         model_wts = model.state_dict()

# #         for (tw,mw) in zip(checkpoint['model'],model_wts):
# #     #         tmp = mw.split('.')

# #             if tw == mw:
# #                 print(f'Loading weight {tw} to {mw}')
# #                 model_wts[mw] = checkpoint['model'][tw]

# #         model.load_state_dict(model_wts)

    else:
        model = DCStudentNet(args).to(args.device)
        model.load_state_dict(checkpoint['model'])
#         print("kd")
        

    

    return model


def run_model(args, model,data_loader):

    model.eval()
    reconstructions = defaultdict(list)

    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            if args.dataset_type == 'knee' :
                input, input_kspace, target,fnames = data
            else:
                input, input_kspace, target,mask,fnames,slices = data

            
            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)
            
            input = input.float()
            target = target.float()
            mask = mask.float()

            recons = model(input,input_kspace,mask)[-1]
            #recons = model(input.float())[-1]

            recons = recons.to('cpu').squeeze(1)
#             print (recons[0].shape)
#             print (recons.dtype)

            #if args.dataset_type == 'cardiac':
            #    recons = recons[:,5:155,5:155]

            if args.dataset_type == 'knee':
                for i in range(recons.shape[0]):
                    recons[i] = recons[i] 
                    reconstructions[fnames[i]].append(recons[i].numpy())
                

            else :
                for i in range(recons.shape[0]):
                    recons[i] = recons[i] 
                    reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))
#                     print(fnames[i])
                

#     print (reconstructions.keys())

#     if args.dataset_type == 'knee':
#         reconstructions = {
#             fname: np.stack([pred for _, pred in sorted(slice_preds)])
#             for fname, slice_preds in reconstructions.items()
#         }
        
#     else:
#         reconstructions = {
#             fname: np.stack([pred for pred in sorted(slice_preds)])
#             for fname, slice_preds in reconstructions.items()
#         }
    if args.dataset_type == 'knee':

        reconstructions = {
            fname: np.stack([pred for pred in sorted(slice_preds)])
            for fname, slice_preds in reconstructions.items()
        }

    else:
         reconstructions = {
             fname: np.stack([pred for _, pred in sorted(slice_preds)])
             for fname, slice_preds in reconstructions.items()
         }

#     print(reconstructions.items())
#     print(reconstructions['1.h5'])

    return reconstructions


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.model_type,args.checkpoint)

    reconstructions = run_model(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='undersampling mask path')
    #parser.add_argument('--data_consistency',action='store_true')
    parser.add_argument('--model_type',type=str,help='model type teacher student')
    parser.add_argument('--mask_type',type=str,help='us mask path')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
