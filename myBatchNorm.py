import torch
from qnn import torchFixpoint

class MyBatchnorm1d(torch.nn.Module):
    def __init__(self,num_features,momentum=0.1):
        '''
        自定义的batchnorm
        :param num_features:
        :param momentum: 动量系数,大于等于0小于1,表示保留原来变量值的比例，与标准库torch.nn.Batchnorm1d
                         当取None时，采用简单的取平均的方式计算running_mean和running_var
        '''
        super(MyBatchnorm1d,self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_features).float())
        self.bias = torch.nn.Parameter(torch.zeros(num_features).float())
        #register_buffer相当于requires_grad=False的Parameter，所以两种方法都可以
        #方法一
        self.register_buffer('running_mean',torch.zeros(num_features))
        self.register_buffer('running_var',torch.zeros(num_features))
        self.register_buffer('num_batches_tracked',torch.tensor(0))
        #方法二
        # self.running_mean = torch.nn.Parameter(torch.zeros(num_features),requires_grad=False)
        # self.running_var = torch.nn.Parameter(torch.ones(num_features),requires_grad=False)
        # self.num_batches_tracked = torch.nn.Parameter(torch.tensor(0),requires_grad=False)
 
        self.momentum = momentum
 
    def forward(self,x):
        if self.training: #训练模型
            #数据是二维的情况下，可以这么处理，其他维的时候不是这样的，但原理都一样。
            mean_bn = x.mean(0, keepdim=True).squeeze(0) #相当于x.mean(0, keepdim=False)
            var_bn = x.var(0, keepdim=True).squeeze(0) #相当于x.var(0, keepdim=False)
 
            if self.momentum is not None:
                self.running_mean.mul_((1 - self.momentum))
                self.running_mean.add_(self.momentum * mean_bn.data)
                self.running_var.mul_((1 - self.momentum))
                self.running_var.add_(self.momentum * var_bn.data)
            else:  #直接取平均,以下是公式变形，即 m_new = (m_old*n + new_value)/(n+1)
                self.running_mean = self.running_mean+(mean_bn.data-self.running_mean)/(self.num_batches_tracked+1)
                self.running_var = self.running_var+(var_bn.data-self.running_var)/(self.num_batches_tracked+1)
            self.num_batches_tracked += 1
        else: #eval模式
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)
 
        eps = 1e-5
        x_normalized = (x - mean_bn) / torch.sqrt(var_bn + eps)
        results = self.weight * x_normalized + self.bias
        return results


class MyBatchnorm2d(torch.nn.Module):
    def __init__(self,num_features,momentum=0.1):
        '''
        自定义的batchnorm
        :param num_features:
        :param momentum: 动量系数,大于等于0小于1,表示保留原来变量值的比例，与标准库torch.nn.Batchnorm1d相同
                         当取None时，采用简单的取平均的方式计算running_mean和running_var
        '''
        super(MyBatchnorm2d,self).__init__()
        shape = (1, num_features, 1, 1)
        self.weight = torch.nn.Parameter(torch.ones(shape).float())
        self.bias = torch.nn.Parameter(torch.zeros(shape).float())
        #register_buffer相当于requires_grad=False的Parameter，所以两种方法都可以
        #方法一
        self.register_buffer('running_mean',torch.zeros(shape))
        self.register_buffer('running_var',torch.zeros(shape))
        self.register_buffer('num_batches_tracked',torch.tensor(0))
        #方法二
        # self.running_mean = torch.nn.Parameter(torch.zeros(num_features),requires_grad=False)
        # self.running_var = torch.nn.Parameter(torch.ones(num_features),requires_grad=False)
        # self.num_batches_tracked = torch.nn.Parameter(torch.tensor(0),requires_grad=False)
 
        self.momentum = momentum
 
    def forward(self,x):
        if self.training: #训练模型
           # mean_bn = x.mean(0, keepdim=True) 
           # var_bn = x.var(0, keepdim=True) 
            mean_bn = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var_bn = ((x - mean_bn) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            
            if self.momentum is not None:
                self.running_mean.mul_((1 - self.momentum))
                self.running_mean.add_(self.momentum * mean_bn.data)
                self.running_var.mul_((1 - self.momentum))
                self.running_var.add_(self.momentum * var_bn.data)
            else:  #直接取平均,以下是公式变形，即 m_new = (m_old*n + new_value)/(n+1)
                self.running_mean = self.running_mean+(mean_bn.data-self.running_mean)/(self.num_batches_tracked+1)
                self.running_var = self.running_var+(var_bn.data-self.running_var)/(self.num_batches_tracked+1)
            self.num_batches_tracked += 1
        else: #eval模式
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)
 
        eps = 1e-5
        x_normalized = (x - mean_bn) / torch.sqrt(var_bn + eps)
        results = self.weight * x_normalized + self.bias
        return results


class MyQBatchnorm2d(torch.nn.Module):
    def __init__(self,num_features,  qlistm=None, qlistb=None):
        '''
        quantized batchnormalization 
        :param num_features:
        :param momentum: 动量系数,大于等于0小于1,表示保留原来变量值的比例，与标准库torch.nn.Batchnorm1d相同
                         当取None时，采用简单的取平均的方式计算running_mean和running_var
        '''
        super(MyQBatchnorm2d,self).__init__()
        shape = (1, num_features, 1, 1)
        self.weight = torch.nn.Parameter(torch.ones(shape).float())
        self.bias = torch.nn.Parameter(torch.zeros(shape).float())
        self.qlistm = qlistm
        self.qlistb = qlistb
        #register_buffer相当于requires_grad=False的Parameter，所以两种方法都可以
        #方法一
        self.register_buffer('running_mean',torch.zeros(shape))
        self.register_buffer('running_var',torch.zeros(shape))
        self.register_buffer('num_batches_tracked',torch.tensor(0))
        #方法二
        # self.running_mean = torch.nn.Parameter(torch.zeros(num_features),requires_grad=False)
        # self.running_var = torch.nn.Parameter(torch.ones(num_features),requires_grad=False)
        # self.num_batches_tracked = torch.nn.Parameter(torch.tensor(0),requires_grad=False)
        momentum=0.1
        self.momentum = momentum
 
    def forward(self,x):
        eps = 1e-5
        if self.training: #训练模型
           # mean_bn = x.mean(0, keepdim=True) 
           # var_bn = x.var(0, keepdim=True) 
            mean_bn = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var_bn = ((x - mean_bn) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            
            if self.momentum is not None:
                self.running_mean.mul_((1 - self.momentum))
                self.running_mean.add_(self.momentum * mean_bn.data)
                self.running_var.mul_((1 - self.momentum))
                self.running_var.add_(self.momentum * var_bn.data)
            else:  #直接取平均,以下是公式变形，即 m_new = (m_old*n + new_value)/(n+1)
                self.running_mean = self.running_mean+(mean_bn.data-self.running_mean)/(self.num_batches_tracked+1)
                self.running_var = self.running_var+(var_bn.data-self.running_var)/(self.num_batches_tracked+1)
            self.num_batches_tracked += 1
            x_normalized = (x - mean_bn) / torch.sqrt(var_bn + eps)
            results = self.weight * x_normalized + self.bias
        else: #eval模式
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)
            multiplier =  self.weight / torch.sqrt(var_bn + eps) 
            bnbias = - mean_bn * self.weight / torch.sqrt(var_bn + eps) + self.bias
            multiplier = torchFixpoint(multiplier, self.qlistm)  
            bnbias = torchFixpoint(bnbias, self.qlistb)
            results = multiplier * x + bnbias

        return results
