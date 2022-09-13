import torch
from torch.nn import Module, Linear, Sigmoid, MaxPool2d, ReLU, ConvTranspose2d, Parameter

class Do_Module(Module):
    """  Cross channel region-level attention (CCRA) mechanism """
    def __init__(self):
        super(Do_Module, self).__init__()
        self.MaxPool = MaxPool2d(kernel_size=2, stride=2, padding=(0,1))
        self.Linear_1 = Linear(3*3,9)
        self.Linear_2 = Linear(3 * 3, 9)
        self.Linear_3 = Linear(3 * 3, 9)
        self.Linear_4 = Linear(3 * 3, 9)
        self.Linear_5 = Linear(3 * 3, 9)
        self.Linear_6 = Linear(3 * 3, 9)
        self.Linear_7 = Linear(3 * 3, 9)
        self.Linear_8 = Linear(3 * 3, 9)
        self.Linear_9 = Linear(3 * 3, 9)
        self.Linear_10 = Linear(3 * 3, 9)
        self.Relu = ReLU(inplace=True)
        self.Sigmoid = Sigmoid()
        self.ConvTranspose2d_1 = ConvTranspose2d(320, 320, kernel_size= (2,1), stride=2)
        self.ConvTranspose2d_2 = ConvTranspose2d(320, 320, kernel_size=(2, 1), stride=2)
        self.ConvTranspose2d_3 = ConvTranspose2d(320, 320, kernel_size=(2, 1), stride=2)
        self.ConvTranspose2d_4 = ConvTranspose2d(320, 320, kernel_size=(2, 1), stride=2)
        self.ConvTranspose2d_5 = ConvTranspose2d(320, 320, kernel_size=(2, 1), stride=2)
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X D X H X W)
            returns :
                out : region_attention value + input feature
        """
        m_batchsize, channels, depth, height, width = x.size()
        temp_X = x.permute(2, 0, 1, 3, 4)
        temp = torch.zeros(temp_X.shape)
        for i in range(depth):
            domain_X = self.MaxPool(temp_X[i])
            domain_m_batchsize, domain_channels, domain_depth, domain_width = domain_X.size()
            domain_query = domain_X.view(domain_m_batchsize, -1, domain_depth*domain_width)
            if i == 0:
                domain_key = self.Sigmoid(self.Linear_2(self.Relu(self.Linear_1(domain_query))))
            elif i == 1:
                domain_key = self.Sigmoid(self.Linear_4(self.Relu(self.Linear_3(domain_query))))
            elif i == 2:
                domain_key = self.Sigmoid(self.Linear_6(self.Relu(self.Linear_5(domain_query))))
            elif i == 3:
                domain_key = self.Sigmoid(self.Linear_8(self.Relu(self.Linear_7(domain_query))))
            elif i == 4:
                domain_key = self.Sigmoid(self.Linear_10(self.Relu(self.Linear_9(domain_query))))
            energy = torch.zeros(domain_query.shape)
            for j in range(domain_m_batchsize):
                for k in range(domain_channels):
                    energy[j][k] = torch.dot(domain_query[j][k],domain_key[j][k])
            domain_value = torch.add(energy.cuda() , domain_query)
            if i == 0:
                temp[i] = self.ConvTranspose2d_1(domain_value.view(domain_m_batchsize, domain_channels, domain_depth, domain_width))
            elif i == 1:
                temp[i] = self.ConvTranspose2d_2(domain_value.view(domain_m_batchsize, domain_channels, domain_depth, domain_width))
            elif i == 2:
                temp[i] = self.ConvTranspose2d_3(domain_value.view(domain_m_batchsize, domain_channels, domain_depth, domain_width))
            elif i == 3:
                temp[i] = self.ConvTranspose2d_4(domain_value.view(domain_m_batchsize, domain_channels, domain_depth, domain_width))
            elif i == 4:
                temp[i] = self.ConvTranspose2d_5(domain_value.view(domain_m_batchsize, domain_channels, domain_depth, domain_width))
        out = temp.permute(1, 2, 0, 3, 4).cuda()
        out = self.gamma * out + x
        return out