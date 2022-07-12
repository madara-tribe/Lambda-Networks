import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import double_conv, LambdaConv
start_fm = 64

class UNet(nn.Module):
    
    def __init__(self, inc, num_cls):
        super(UNet, self).__init__()
        
        # Input 128x128x1
        
        #Contracting Path
        lambda_heads = 4
        #(Double) Convolution 1        
        self.double_conv1 = double_conv(inc, start_fm)
        #Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 2
        self.double_conv2 = double_conv(start_fm, start_fm * 2)
        #Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 3
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4)
        #Max Pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.lambda_conv3 = LambdaConv(start_fm * 4, start_fm * 4, heads=lambda_heads, k=16, u=1)
        
        #Convolution 4
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8)
        #Max Pooling 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.lambda_conv4 = LambdaConv(start_fm * 8, start_fm * 8, heads=lambda_heads, k=16, u=1)
        
        
        #Convolution 5
        self.double_conv5 = double_conv(start_fm * 8, start_fm * 16)
        
        #Transposed Convolution 4
        self.t_conv4 = nn.ConvTranspose2d(start_fm * 16, start_fm * 8, 2, 2)
        # Expanding Path Convolution 4 
        self.ex_double_conv4 = double_conv(start_fm * 16, start_fm * 8)
        
        #Transposed Convolution 3
        self.t_conv3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2)
        #Convolution 3
        self.ex_double_conv3 = double_conv(start_fm * 8, start_fm * 4)
        
        #Transposed Convolution 2
        self.t_conv2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2)
        #Convolution 2
        self.ex_double_conv2 = double_conv(start_fm * 4, start_fm * 2)
        
        #Transposed Convolution 1
        self.t_conv1 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)
        #Convolution 1
        self.ex_double_conv1 = double_conv(start_fm * 2, start_fm)
        
        # One by One Conv
        self.one_by_one = nn.Conv2d(start_fm, num_cls, kernel_size=3, padding=1)
        self.final_act = nn.Sigmoid()
        
        
    def forward(self, inputs):
        # Contracting Path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        
        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        maxpool3 = self.lambda_conv3(maxpool3) # [1, 256, 16, 16]
        
        conv4 = self.double_conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4) # [1, 512, 8, 8]
        maxpool4 = self.lambda_conv4(maxpool4) # [1, 512, 8, 8]
        
        # Bottom
        conv5 = self.double_conv5(maxpool4)
        
        # Expanding Path
        t_conv4 = self.t_conv4(conv5)
        cat4 = torch.cat([conv4 ,t_conv4], 1)
        ex_conv4 = self.ex_double_conv4(cat4)
        
        t_conv3 = self.t_conv3(ex_conv4)
        cat3 = torch.cat([conv3 ,t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        t_conv2 = self.t_conv2(ex_conv3)
        cat2 = torch.cat([conv2 ,t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)
        
        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([conv1 ,t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)
        
        one_by_one = self.one_by_one(ex_conv1)
        x = self.final_act(one_by_one)
        return x
