import torch
import torch.nn as nn
import torch.utils

class AFIR(nn.Module):
    def __init__(self, M, D):
        super(AFIR, self).__init__()
        self.real = nn.Conv1d(1, 1, M, padding=int((M-1)/2), bias=False).double()
        self.imag = nn.Conv1d(1, 1, M, padding=int((M-1)/2), bias=False).double()
        self.real.weight.data.fill_(0.0)
        self.imag.weight.data.fill_(0.0)
        self.real.weight.data[0, 0, int((M-1)/2)+D] = 1.0
    def forward(self, x):
        r1 = self.real(x[:,0].view(1,1,-1))
        r2 = self.imag(x[:,1].view(1,1,-1))
        i1 = self.real(x[:,1].view(1,1,-1))
        i2 = self.imag(x[:,0].view(1,1,-1))
        return torch.cat((r1-r2, i1+i2), dim=1)

# class Delay(AFIR):
#     def __init__(self,M,D):
#         super(Delay,self).__init__(M,D)
#         self.real.weight.requires_grad=False
#         self.imag.weight.requires_grad=False
#         self.imag.weight.data[0, 0, int((M-1)/2)+D] = 1.0
#     def forward(self, x):
#             r = self.real(x[:,0].view(1,1,-1))
#             i = self.imag(x[:,1].view(1,1,-1))
#             return torch.cat((r, i), dim=1)
class Delay(nn.Module):
    def __init__(self, M):
        super(Delay, self).__init__()
        self.M = M
        self.op = nn.Sequential(
            nn.ConstantPad1d(abs(M),0)
        )
    def forward(self, x):
        return self.op(x)[:,:,:x.shape[2]] if self.M>0 else self.op(x)[:,:,-x.shape[2]:] 

class Prod_cmp(nn.Module):
    def __init__(self):
        super(Prod_cmp, self).__init__()
    def forward(self, inp1, inp2):
        r1 = inp1[:,0].view(1,1,-1)*inp2[:,0].view(1,1,-1)
        r2 = inp1[:,1].view(1,1,-1)*inp2[:,1].view(1,1,-1)
        i1 = inp1[:,1].view(1,1,-1)*inp2[:,0].view(1,1,-1)
        i2 = inp1[:,0].view(1,1,-1)*inp2[:,1].view(1,1,-1)
        return torch.cat((r1-r2, i1+i2), dim=1)



class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, x):
        out = x.norm(dim=1, keepdim=True)
        return out

class Polynomial(nn.Module):
    def __init__(self, Poly_order,passthrough=False):
        super(Polynomial, self).__init__()
        self.order = Poly_order
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weights = nn.Parameter(torch.zeros((2, Poly_order), device=device, dtype=torch.float64), requires_grad=True)
        if passthrough:
            self.weights.data[0, 1] = 1
#         else:
#             torch.linspace(0,1,Poly_order,out=self.weights[0,:],device=device,requires_grad=True)
        self.Abs = ABS()
    def forward(self, x):
        out = torch.zeros_like(x)
        x = self.Abs(x).view(1, -1)
        for i in range(self.order):
            out[:, 0] += self.weights[0, i]*torch.pow(x,i)
            out[:, 1] += self.weights[1, i]*torch.pow(x,i)
        return out