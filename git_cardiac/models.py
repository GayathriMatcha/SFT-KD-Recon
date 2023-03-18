import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import math

    
class DataConsistencyLayer(nn.Module):

    def __init__(self):

        super(DataConsistencyLayer,self).__init__()

    def forward(self,predicted_img,us_kspace,us_mask):
#         print(f'predicted_img.shape1: {predicted_img.shape}')
        predicted_img = predicted_img[:,:,5:-5,5 :-5]
#         print(predicted_img.shape)

        kspace_predicted_img = torch.fft.fft2(predicted_img,norm = "ortho")
        
#         print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,us_mask.shape)

        us_kspace_complex = us_kspace[:,:,:,:,0]+us_kspace[:,:,:,:,1]*1j

        updated_kspace1  = us_mask * us_kspace_complex

        updated_kspace2  = (1 - us_mask) * kspace_predicted_img

        updated_kspace = updated_kspace1 + updated_kspace2

        updated_img  = torch.fft.ifft2(updated_kspace,norm = "ortho")

        updated_img = torch.view_as_real(updated_img)
        
        update_img_abs = updated_img[:,:,:,:,0] # taking real part only, change done on Sep 18 '19 bcos taking abs till bring in the distortion due to imag part also. this was verified was done by simple experiment on FFT, mask and IFFT
#         update_img_abs  = np.pad(update_img_abs,(5,5),'constant',constant_values=(0,0))
#         update_img_abs  = torch.nn.functional.pad(update_img_abs, (5,5,5,5), 'constant', value=0)
#         print(f'update_img_abs.shapef1: {update_img_abs.shape}')
        
        return update_img_abs.float()


class TeacherNet(nn.Module):
    
    def __init__(self):
        super(TeacherNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = self.conv5(x4)
        
        return x1,x2,x3,x4,x5


class DCTeacherNet(nn.Module):

    def __init__(self,args):

        super(DCTeacherNet,self).__init__()

        self.tcascade1 = TeacherNet()
        self.tcascade2 = TeacherNet()
        self.tcascade3 = TeacherNet()
        self.tcascade4 = TeacherNet()
        self.tcascade5 = TeacherNet()


        self.tdc = DataConsistencyLayer()
        self.pad2d = nn.ConstantPad2d(5,0)

    def forward(self,x,x_k,us_mask):
        
        x1 = self.tcascade1(self.pad2d(x))      
        x1_dc = self.tdc(x1[-1],x_k,us_mask)

        x2 = self.tcascade2(self.pad2d(x1_dc))
        x2_dc = self.tdc(x2[-1],x_k,us_mask)

        x3 = self.tcascade3(self.pad2d(x2_dc))
        x3_dc = self.tdc(x3[-1],x_k,us_mask)

        x4 = self.tcascade4(self.pad2d(x3_dc))
        x4_dc = self.tdc(x4[-1],x_k,us_mask)

        x5 = self.tcascade5(self.pad2d(x4_dc))
        x5_dc = self.tdc(x5[-1],x_k,us_mask)

        return x1,x2,x3,x4,x5,x5_dc


class StudentNet(nn.Module):
    
    def __init__(self):
        super(StudentNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)
        
        return x1,x2,x3


class DCStudentNet(nn.Module):

    def __init__(self,args):

        super(DCStudentNet,self).__init__()

        self.scascade1 = StudentNet()
        self.scascade2 = StudentNet()
        self.scascade3 = StudentNet()
        self.scascade4 = StudentNet()
        self.scascade5 = StudentNet()

        self.sdc = DataConsistencyLayer()#sus_mask
        self.pad2d = nn.ConstantPad2d(5,0)

    def forward(self,x,x_k,us_mask):

        x1 = self.scascade1(self.pad2d(x))  
        x1_dc = self.sdc(x1[-1],x_k,us_mask)

        x2 = self.scascade2(self.pad2d(x1_dc))
        x2_dc = self.sdc(x2[-1],x_k,us_mask)

        x3 = self.scascade3(self.pad2d(x2_dc))
        x3_dc = self.sdc(x3[-1],x_k,us_mask)

        x4 = self.scascade4(self.pad2d(x3_dc))
        x4_dc = self.sdc(x4[-1],x_k,us_mask)

        x5 = self.scascade5(self.pad2d(x4_dc))
        x5_dc = self.sdc(x5[-1],x_k,us_mask)

        return x1,x2,x3,x4,x5,x5_dc

############################################################*SFTN_TEACHER*#

class DCTeacherNetSFTN(nn.Module):

    def __init__(self,args):

        super(DCTeacherNetSFTN,self).__init__()
        

        self.tcascade1 = TeacherNet()
        self.tcascade2 = TeacherNet()
        self.tcascade3 = TeacherNet()
        self.tcascade4 = TeacherNet()
        self.tcascade5 = TeacherNet()
        
        self.scascade1 = StudentNet()
        self.scascade2 = StudentNet()
        self.scascade3 = StudentNet()
        self.scascade4 = StudentNet()
        self.scascade5 = StudentNet()

        self.tdc = DataConsistencyLayer()
        self.sdc = DataConsistencyLayer()
        self.pad2d = nn.ConstantPad2d(5,0)

    def forward(self,x,x_k,us_mask):
        if self.training:

            x1 = self.tcascade1(self.pad2d(x))  
            x1_dc = self.tdc(x1[-1],x_k,us_mask)

            x2 = self.tcascade2(self.pad2d(x1_dc))
            x2_dc = self.tdc(x2[-1],x_k,us_mask)

            x3 = self.tcascade3(self.pad2d(x2_dc))
            x3_dc = self.tdc(x3[-1],x_k,us_mask)

            x4 = self.tcascade4(self.pad2d(x3_dc))
            x4_dc = self.tdc(x4[-1],x_k,us_mask)

            x5 = self.tcascade5(self.pad2d(x4_dc))
            x5_dc = self.tdc(x5[-1],x_k,us_mask)

    ##################################################
            x_s1 = self.scascade2(self.pad2d(x1_dc))
            x_s1dc = self.sdc(x_s1[-1],x_k,us_mask)

            x_s1 = self.scascade3(self.pad2d(x_s1dc))
            x_s1dc = self.sdc(x_s1[-1],x_k,us_mask)

            x_s1 = self.scascade4(self.pad2d(x_s1dc))
            x_s1dc = self.sdc(x_s1[-1],x_k,us_mask)

            x_s1 = self.scascade5(self.pad2d(x_s1dc))
            x_s1dc = self.sdc(x_s1[-1],x_k,us_mask)

    ####################################################        
            x_s2 = self.scascade3(self.pad2d(x2_dc))
            x_s2dc = self.sdc(x_s2[-1],x_k,us_mask)

            x_s2 = self.scascade4(self.pad2d(x_s2dc))
            x_s2dc = self.sdc(x_s2[-1],x_k,us_mask)

            x_s2 = self.scascade5(self.pad2d(x_s2dc))
            x_s2dc = self.sdc(x_s2[-1],x_k,us_mask)

    #######################################################

            x_s3 = self.scascade4(self.pad2d(x3_dc))
            x_s3dc = self.sdc(x_s3[-1],x_k,us_mask)

            x_s3 = self.scascade5(self.pad2d(x_s3dc))
            x_s3dc = self.sdc(x_s3[-1],x_k,us_mask)

    #########################################################       

            x_s4 = self.scascade5(self.pad2d(x4_dc))
            x_s4dc = self.sdc(x_s4[-1],x_k,us_mask)
            
            op = x_s1dc, x_s2dc, x_s3dc, x_s4dc, x5_dc
            
        else: #Eval mode
            
            x1 = self.tcascade1(self.pad2d(x))      
            x1_dc = self.tdc(x1[-1],x_k,us_mask)

            x2 = self.tcascade2(self.pad2d(x1_dc))
            x2_dc = self.tdc(x2[-1],x_k,us_mask)

            x3 = self.tcascade3(self.pad2d(x2_dc))
            x3_dc = self.tdc(x3[-1],x_k,us_mask)

            x4 = self.tcascade4(self.pad2d(x3_dc))
            x4_dc = self.tdc(x4[-1],x_k,us_mask)

            x5 = self.tcascade5(self.pad2d(x4_dc))
            x5_dc = self.tdc(x5[-1],x_k,us_mask)
            
            op = x1,x2,x3,x4,x5,x5_dc
            
        
        return op
    
############################################################*INTERACTIVE_KD*#
    
class IKD(nn.Module):
    def __init__(self) -> None:
        super(IKD,self).__init__()
        
        self.tcascades = nn.ModuleList([TeacherNet() for _ in range(5)])

        for item in self.tcascades:
            for p in item.parameters():
                p.requires_grad = False

        self.scascades = nn.ModuleList([StudentNet() for _ in range(5)])

        self.dc = DataConsistencyLayer()

    def forward(self,x,x_k,us_mask,cfg=[True]*5):
        
        if self.training: # Training mode
#             print(f'cfg = {cfg}')

            if cfg[0]:
                x = self.scascades[0](x)

            else:
                x = self.tcascades[0](x)

#             x = self.scascades[0](x)
            x = self.dc(x[-1],x_k,us_mask)

            if cfg[1]: # Select student
                x = self.scascades[1](x)

            else:
                x = self.tcascades[1](x)

            x = self.dc(x[-1],x_k,us_mask)


            if cfg[2]: # Select student
                x = self.scascades[2](x)

            else:
                x = self.tcascades[2](x)

            x = self.dc(x[-1],x_k,us_mask)


            if cfg[3]: # Select student
                x = self.scascades[3](x)

            else:
                x = self.tcascades[3](x)

            x = self.dc(x[-1],x_k,us_mask)

            if cfg[4]: # Select student
                x = self.scascades[4](x)

            else:
                x = self.tcascades[4](x)

            x = self.dc(x[-1],x_k,us_mask)


        else: # Eval mode
            x = self.scascades[0](x)  
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[1](x)
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[2](x)
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[3](x)
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[4](x)
            x = self.dc(x[-1],x_k,us_mask)

        return x
    
class IKD(nn.Module):
    def __init__(self) -> None:
        super(IKD,self).__init__()
        
        self.tcascades = nn.ModuleList([TeacherNet() for _ in range(5)])

        for item in self.tcascades:
            for p in item.parameters():
                p.requires_grad = False

        self.scascades = nn.ModuleList([StudentNet() for _ in range(5)])

        self.dc = DataConsistencyLayer()
        
        self.tcascade1 = TeacherNet()
        self.tcascade2 = TeacherNet()
        self.tcascade3 = TeacherNet()
        self.tcascade4 = TeacherNet()
        self.tcascade5 = TeacherNet()
        
        self.scascade1 = StudentNet()
        self.scascade2 = StudentNet()
        self.scascade3 = StudentNet()
        self.scascade4 = StudentNet()
        self.scascade5 = StudentNet()

        self.tdc = DataConsistencyLayer()
        self.sdc = DataConsistencyLayer()
        self.pad2d = nn.ConstantPad2d(5,0)

    def forward(self,x,x_k,us_mask,cfg=[True]*5):
        
        if self.training: # Training mode
#             print(f'cfg = {cfg}')

            if cfg[0]:
                x = self.scascades[0](x)

            else:
                x = self.tcascades[0](x)

#             x = self.scascades[0](x)
            x = self.dc(x[-1],x_k,us_mask)

            if cfg[1]: # Select student
                x = self.scascades[1](x)

            else:
                x = self.tcascades[1](x)

            x = self.dc(x[-1],x_k,us_mask)


            if cfg[2]: # Select student
                x = self.scascades[2](x)

            else:
                x = self.tcascades[2](x)

            x = self.dc(x[-1],x_k,us_mask)


            if cfg[3]: # Select student
                x = self.scascades[3](x)

            else:
                x = self.tcascades[3](x)

            x = self.dc(x[-1],x_k,us_mask)

            if cfg[4]: # Select student
                x = self.scascades[4](x)

            else:
                x = self.tcascades[4](x)

            x = self.dc(x[-1],x_k,us_mask)


        else: # Eval mode
            x = self.scascades[0](x)  
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[1](x)
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[2](x)
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[3](x)
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[4](x)
            x = self.dc(x[-1],x_k,us_mask)

        return x