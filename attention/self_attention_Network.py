import torch
from torch.nn import Module, Conv2d, Conv3d, Parameter, Softmax, LeakyReLU

class SA_Module(Module):
    """ Position self-attention (PSA) mechanism"""
    def __init__(self):
        super(SA_Module, self).__init__()

        self.weight_W = Parameter(torch.rand(2*320,1))
        self.leakyrelu = LeakyReLU(negative_slope=0.01, inplace=True)
        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X D X H X W)
            returns :
                out : attention value + input feature
                attention: B X (DXHxW) X (DXHxW)
        """
        m_batchsize, channels, depth, height, width = x.size()
        proj_temp = x.view(m_batchsize, -1, depth*width*height)  
        energy = torch.zeros(m_batchsize, depth*width*height, depth*width*height) 
        temp_d = []
        for i in range(energy.shape[0]):
            temp = proj_temp[i].permute(1, 0) 
            for j in range(depth*width*height):
                temp_a = torch.zeros(temp.shape)
                temp_a.copy_(temp[j])
                temp_b = torch.cat((temp, temp_a.cuda()), 1) 
                if j == 0:
                    temp_c = torch.matmul(temp_b,self.weight_W).permute(1,0)
                else:
                    temp_c = torch.cat((temp_c,torch.matmul(temp_b,self.weight_W).permute(1,0)),0)
            temp_d.append(temp_c)
        if energy.shape[0] > 1:
            energy = torch.cat((temp_d[0].unsqueeze(0), temp_d[1].unsqueeze(0)), 0)
        else:
            energy = temp_d[0].unsqueeze(0)
        attention = self.softmax(self.leakyrelu(energy))  
        out = torch.bmm(proj_temp, attention.permute(0, 2, 1).cuda()) 
        out = out + proj_temp
        out = out.view(m_batchsize, channels, depth, height, width)
        return out