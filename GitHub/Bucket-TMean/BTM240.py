import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import random
from torch import optim
from torch.autograd import Variable


#random.seed(0)
#np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

device = 'cuda:3'
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])
testset = tv.datasets.CIFAR10('~/data1/',train=False,download=True,transform=transform)#10000
trainset = tv.datasets.CIFAR10(root='~/data1/',train = True,download=True,transform=transform)#50000

B = 1
Node = 10
Label = 10
batchsize = 50
mmm = 0.9
pr_c = 10
xi_s_list = [0]
xi_w_list = [0]
xi_s = xi_s_list[0]
xi_w = xi_w_list[0]
BG_list = [10,5,3,2]
BGroup = BG_list[0]
stepsize = 0.002
penal = 0.001
regu = 0.01
beta = 0.2
tau = 25
It = 10
iteration = 30000
count_point = 100#画图要多少个点
count_iter = int(iteration/count_point)#每多少个epoch计算一次acc和H
criterion  = nn.CrossEntropyLoss()#定义交叉熵损失函数


#测试集数据
data_test_temp = []
label_test_temp = []
for i in range(len(testset)):
    data_test_temp.append(testset[i][0])
    label_test_temp.append(testset[i][1])

data_test = []
label_test = []
for i in range(int(len(data_test_temp)/batchsize)):
    data_batchtemp = []
    label_batchtemp = []
    for bs in range(batchsize):
        data_batchtemp.append(data_test_temp[i*batchsize+bs].numpy().tolist())
        label_batchtemp.append(label_test_temp[i*batchsize+bs])
    data_test.append(torch.tensor(data_batchtemp, device=device, requires_grad=False))
    label_test.append(torch.tensor(np.array(label_batchtemp), device=device, requires_grad=False))


#数据在生成时已经做了混合打乱，程序主体中每一轮迭代相当于做无放回抽样
data_len = len(trainset)
data_temp = [[] for _ in range(Label)]#10
label_temp = [[] for _ in range(Label)]
for i in range(data_len):
    label_temp[trainset[i][1]].append(trainset[i][1])#5000
    data_temp[trainset[i][1]].append(trainset[i][0])

#——————iid数据集——————
#先从每个标签中随机抽样放进每个节点里
data_iid_temp = [[] for _ in range(Node)]
label_iid_temp = [[] for _ in range(Node)]
rand_index = list(range(len(data_temp[0])))#每个标签下的数据个数，默认5000
random.shuffle(rand_index)
for k in range(Node):
    for l in range(Label):
        for j in range(data_len // (Label*Node)):#250
            data_iid_temp[k].append(data_temp[l][rand_index[j + k*(data_len//(Label*Node))]].to(device))
            label_iid_temp[k].append(label_temp[l][rand_index[j + k*(data_len//(Label*Node))]])
#打乱，按batchsize分组打包
data_iid = [[] for _ in range(Node)]
label_iid = [[] for _ in range(Node)]
for k in range(Node):
    rand_index = list(range(len(data_iid_temp[0])))#依次取随机index的十个段
    random.shuffle(rand_index)
    for j in range(len(data_iid_temp[0])//batchsize):#组数
        data_batchtemp = []
        label_batchtemp = []
        for bs in range(batchsize):
            data_batchtemp.append(data_iid_temp[k][rand_index[j*batchsize+bs]][None])
            label_batchtemp.append(label_iid_temp[k][rand_index[j*batchsize+bs]])
        data_iid[k].append(torch.cat(data_batchtemp, dim=0))
        label_iid[k].append(torch.from_numpy(np.array(label_batchtemp)).to(device))

#——————noniid数据集——————
#从每个标签中随机抽样放进每个节点里
data_noniid_temp = [[] for _ in range(Node)]
label_noniid_temp = [[] for _ in range(Node)]
#先放一半
for k in range(Node):
    for j in range(len(data_temp[0])//2):#2500
        data_noniid_temp[k].append(data_temp[k][j].to(device))
        label_noniid_temp[k].append(label_temp[k][j])
    #再放一半
data_noniid_mix = []
label_noniid_mix = []
for k in range(Node):
    for j in range(len(data_temp[0])//2):              #先混合
        data_noniid_mix.append(data_temp[k][len(data_temp)//2+j])
        label_noniid_mix.append(label_temp[k][len(data_temp)//2+j])
rand_index = list(range(len(data_noniid_mix)))      #打乱
random.shuffle(rand_index)
for k in range(10):
    for i in range(len(data_temp[0])//2):               #分配
        data_noniid_temp[k].append(data_noniid_mix[rand_index[(len(data_temp[0])//2)*k + i]].to(device))
        label_noniid_temp[k].append(label_noniid_mix[rand_index[(len(label_temp[0])//2)*k + i]])

#打乱，按batchsize分组打包
data_noniid = [[] for _ in range(Node)]
label_noniid = [[] for _ in range(Node)]
for k in range(Node):
    rand_index = list(range(len(data_noniid_temp[0])))#依次取随机index的十个段,5000
    random.shuffle(rand_index)
    for j in range(len(data_noniid_temp[0])//batchsize):#组数
        data_batchtemp = []
        label_batchtemp = []
        for bs in range(batchsize):
            data_batchtemp.append(data_noniid_temp[k][rand_index[j*batchsize+bs]][None])
            label_batchtemp.append(label_noniid_temp[k][rand_index[j*batchsize+bs]])
        data_noniid[k].append(torch.cat(data_batchtemp, dim=0))
        label_noniid[k].append(torch.from_numpy(np.array(label_batchtemp)).to(device))


#Predefine
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = torch.nn.MaxPool2d(kernel_size = 3,stride = 2)
        self.conv2 = torch.nn.Conv2d(64,64,5)
        self.fc1 = torch.nn.Linear(64*4*4,384)
        self.fc2 = torch.nn.Linear(384,192)
        self.fc3 = torch.nn.Linear(192,10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def Accuracy(net, data_test, label_test):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(data_test)):
            images = data_test[i]
            labels = label_test[i]
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1) # 第一个变量不重要，用-代替
            correct +=(predicted == labels).sum()      # 更新正确分类的图片的数量
            total += len(data_test[i])                  # 更新测试图片的数量
    return correct/total


def mean_aggregation(paras):
    return torch.mean(torch.cat(paras, dim=0), dim=0)

def TMeans(paras0, GoodNode):
    paras_line = torch.cat(paras0, dim=1).T
    med_values = paras_line.median(dim=0).values
    med_dist = torch.sub(paras_line,med_values)
    sort_index = med_dist.abs().sort(0).indices
    para_tm = torch.cat([paras_line.gather(0,torch.unsqueeze(sort_index[gn],0))[None] for gn in range(GoodNode)], dim=0).mean(dim=0)
    return para_tm.T


def censor_S():
    paranorm_list = []
    for pl in range(10):
        paranorm_list.append(torch.norm(last0_fc1_w_g-net0.fc1.weight.grad.clone()))
        paranorm_list.append(torch.norm(last0_fc2_w_g-net0.fc2.weight.grad.clone()))
        paranorm_list.append(torch.norm(last0_fc3_w_g-net0.fc3.weight.grad.clone()))
        paranorm_list.append(torch.norm(last0_conv1_w_g-net0.conv1.weight.grad.clone()))
        paranorm_list.append(torch.norm(last0_conv2_w_g-net0.conv2.weight.grad.clone()))
        paranorm_list.append(torch.norm(last0_fc1_b_g-net0.fc1.bias.grad.clone()))
        paranorm_list.append(torch.norm(last0_fc2_b_g-net0.fc2.bias.grad.clone()))
        paranorm_list.append(torch.norm(last0_fc3_b_g-net0.fc3.bias.grad.clone()))
        paranorm_list.append(torch.norm(last0_conv1_b_g-net0.conv1.bias.grad.clone()))
        paranorm_list.append(torch.norm(last0_conv2_b_g-net0.conv2.bias.grad.clone()))
    return sum(paranorm_list)

def censor_W(k):
    paranorm_list = []
    for pl in range(10):
        paranorm_list.append(torch.norm(last_fc1_w_g_N[k]-net[k].fc1.weight.grad.clone()))
        paranorm_list.append(torch.norm(last_fc2_w_g_N[k]-net[k].fc2.weight.grad.clone()))
        paranorm_list.append(torch.norm(last_fc3_w_g_N[k]-net[k].fc3.weight.grad.clone()))
        paranorm_list.append(torch.norm(last_conv1_w_g_N[k]-net[k].conv1.weight.grad.clone()))
        paranorm_list.append(torch.norm(last_conv2_w_g_N[k]-net[k].conv2.weight.grad.clone()))
        paranorm_list.append(torch.norm(last_fc1_b_g_N[k]-net[k].fc1.bias.grad.clone()))
        paranorm_list.append(torch.norm(last_fc2_b_g_N[k]-net[k].fc2.bias.grad.clone()))
        paranorm_list.append(torch.norm(last_fc3_b_g_N[k]-net[k].fc3.bias.grad.clone()))
        paranorm_list.append(torch.norm(last_conv1_b_g_N[k]-net[k].conv1.bias.grad.clone()))
        paranorm_list.append(torch.norm(last_conv2_b_g_N[k]-net[k].conv2.bias.grad.clone()))
    return sum(paranorm_list)


def ParaNorm1(net):
    net_para_list = []
    paranorm_list = []
    for name, para in net.named_parameters():
        net_para_list.append(para)
    for pl in range(len(net_para_list)):
        paranorm_list.append(torch.norm(net_para_list[pl]))
    return sum(paranorm_list)



for pr1 in range(len(xi_s_list)):
    xi_s = xi_s_list[pr1]

    for pr2 in range(len(xi_w_list)):
        xi_w = xi_w_list[pr2]

        for pr3 in range(len(BG_list)):
            BGroup = BG_list[pr3]

            for prb in range(4):
                B = prb+1

                #Initialization
                net0 = Net().to(device)
                net = []
                for k in range(Node):
                    net.append(Net().to(device))
                for k in range(Node):
                    net[k].load_state_dict(net0.state_dict())

                optimizer0 = optim.SGD(net0.parameters(),lr = stepsize,momentum=mmm)
                optimizer = []
                for k in range(Node):
                    optimizer.append(optim.SGD(net[k].parameters(),lr = stepsize,momentum=mmm))

                #开始训练
                HS_sum = 0
                HW_sum = 0
                HS_list = []
                HW_list = []
                HS_sum_list = []
                HS_sign = [1]
                HW_sum_list = []
                HW_sign = [1 for k in range(Node)]
                accuracy_list = []
                loss_list = []
                outputs = [[] for k in range(Node)]
                loss = [[] for k in range(Node)]
                outputs0 = [[] for k in range(Node)]
                loss0 = [[] for k in range(Node)]
                inputs = [[] for k in range(Node)]
                labels = [[] for k in range(Node)]
                dist = [[] for k in range(Node)]
                para_cat = [[] for k in range(Node)]
                last_fc1_w_g_N = [[] for k in range(Node)]
                last_fc2_w_g_N = [[] for k in range(Node)]
                last_fc3_w_g_N = [[] for k in range(Node)]
                last_conv1_w_g_N = [[] for k in range(Node)]
                last_conv2_w_g_N = [[] for k in range(Node)]
                last_fc1_b_g_N = [[] for k in range(Node)]
                last_fc2_b_g_N = [[] for k in range(Node)]
                last_fc3_b_g_N = [[] for k in range(Node)]
                last_conv1_b_g_N = [[] for k in range(Node)]
                last_conv2_b_g_N = [[] for k in range(Node)]
                last_fc1_w_g_B = [[] for k in range(Node)]
                last_fc2_w_g_B = [[] for k in range(Node)]
                last_fc3_w_g_B = [[] for k in range(Node)]
                last_conv1_w_g_B = [[] for k in range(Node)]
                last_conv2_w_g_B = [[] for k in range(Node)]
                last_fc1_b_g_B = [[] for k in range(Node)]
                last_fc2_b_g_B = [[] for k in range(Node)]
                last_fc3_b_g_B = [[] for k in range(Node)]
                last_conv1_b_g_B = [[] for k in range(Node)]
                last_conv2_b_g_B = [[] for k in range(Node)]
                last0_fc1_w_g = []
                last0_fc2_w_g = []
                last0_fc3_w_g = []
                last0_conv1_w_g = []
                last0_conv2_w_g = []
                last0_fc1_b_g = []
                last0_fc2_b_g = []
                last0_fc3_b_g = []
                last0_conv1_b_g = []
                last0_conv2_b_g = []
                fc1_w_size = net0.fc1.weight.data.size()
                fc2_w_size = net0.fc2.weight.data.size()
                fc3_w_size = net0.fc3.weight.data.size()
                conv1_w_size = net0.conv1.weight.data.size()
                conv2_w_size = net0.conv2.weight.data.size()
                fc1_b_size = net0.fc1.bias.data.size()
                fc2_b_size = net0.fc2.bias.data.size()
                fc3_b_size = net0.fc3.bias.data.size()
                conv1_b_size = net0.conv1.bias.data.size()
                conv2_b_size = net0.conv2.bias.data.size()
                fc1_w_len = len(net0.fc1.weight.data.view([-1]))
                fc2_w_len = len(net0.fc2.weight.data.view([-1]))
                fc3_w_len = len(net0.fc3.weight.data.view([-1]))
                conv1_w_len = len(net0.conv1.weight.data.view([-1]))
                conv2_w_len = len(net0.conv2.weight.data.view([-1]))
                fc1_b_len = len(net0.fc1.bias.data.view([-1]))
                fc2_b_len = len(net0.fc2.bias.data.view([-1]))
                fc3_b_len = len(net0.fc3.bias.data.view([-1]))
                conv1_b_len = len(net0.conv1.bias.data.view([-1]))
                conv2_b_len = len(net0.conv2.bias.data.view([-1]))

                for epoch in range(iteration):
                    running_loss = 0
                    rand_i = random.choice(list(range(len(data_iid[0]))))
                    for k in range(Node):
                        inputs[k] = data_noniid[k][rand_i]
                        labels[k] = label_noniid[k][rand_i]
                    optimizer0.zero_grad()
                    for k in range(Node):
                        optimizer[k].zero_grad()
                        outputs[k] = net[k](inputs[k])
                        loss[k] = criterion(outputs[k], labels[k])
                        loss[k].backward()

                    if epoch==0:#clipping需要初始值
                        net0.fc1.weight.grad = mean_aggregation([net[k].fc1.weight.grad.clone()[None] for k in range(Node)])
                        net0.fc2.weight.grad = mean_aggregation([net[k].fc2.weight.grad.clone()[None] for k in range(Node)])
                        net0.fc3.weight.grad = mean_aggregation([net[k].fc3.weight.grad.clone()[None] for k in range(Node)])
                        net0.conv1.weight.grad = mean_aggregation([net[k].conv1.weight.grad.clone()[None] for k in range(Node)])
                        net0.conv2.weight.grad = mean_aggregation([net[k].conv2.weight.grad.clone()[None] for k in range(Node)])
                        net0.fc1.bias.grad = mean_aggregation([net[k].fc1.bias.grad.clone()[None] for k in range(Node)])
                        net0.fc2.bias.grad = mean_aggregation([net[k].fc2.bias.grad.clone()[None] for k in range(Node)])
                        net0.fc3.bias.grad = mean_aggregation([net[k].fc3.bias.grad.clone()[None] for k in range(Node)])
                        net0.conv1.bias.grad = mean_aggregation([net[k].conv1.bias.grad.clone()[None] for k in range(Node)])
                        net0.conv2.bias.grad = mean_aggregation([net[k].conv2.bias.grad.clone()[None] for k in range(Node)])
                        last0_fc1_w_g = net0.fc1.weight.grad.clone()
                        last0_fc2_w_g = net0.fc2.weight.grad.clone()
                        last0_fc3_w_g = net0.fc3.weight.grad.clone()
                        last0_conv1_w_g = net0.conv1.weight.grad.clone()
                        last0_conv2_w_g = net0.conv2.weight.grad.clone()
                        last0_fc1_b_g = net0.fc1.bias.grad.clone()
                        last0_fc2_b_g = net0.fc2.bias.grad.clone()
                        last0_fc3_b_g = net0.fc3.bias.grad.clone()
                        last0_conv1_b_g = net0.conv1.bias.grad.clone()
                        last0_conv2_b_g = net0.conv2.bias.grad.clone()
                        for k in range(Node):
                            last_fc1_w_g_N[k] = net[k].fc1.weight.grad.clone()
                            last_fc2_w_g_N[k] = net[k].fc2.weight.grad.clone()
                            last_fc3_w_g_N[k] = net[k].fc3.weight.grad.clone()
                            last_conv1_w_g_N[k] = net[k].conv1.weight.grad.clone()
                            last_conv2_w_g_N[k] = net[k].conv2.weight.grad.clone()
                            last_fc1_b_g_N[k] = net[k].fc1.bias.grad.clone()
                            last_fc2_b_g_N[k] = net[k].fc2.bias.grad.clone()
                            last_fc3_b_g_N[k] = net[k].fc3.bias.grad.clone()
                            last_conv1_b_g_N[k] = net[k].conv1.bias.grad.clone()
                            last_conv2_b_g_N[k] = net[k].conv2.bias.grad.clone()

                    for k in range(B):
                        if HW_sign[k]==1:#censoring
                            last_fc1_w_g_B[k] = (-1) * torch.stack([net[k].fc1.weight.grad.clone().detach() for k in range(10)],2).sum(axis=2).clone()
                            last_fc2_w_g_B[k] = (-1) * torch.stack([net[k].fc2.weight.grad.clone().detach() for k in range(10)],2).sum(axis=2).clone()
                            last_fc3_w_g_B[k] = (-1) * torch.stack([net[k].fc3.weight.grad.clone().detach() for k in range(10)],2).sum(axis=2).clone()
                            last_conv1_w_g_B[k] = (-1) * torch.stack([net[k].conv1.weight.grad.clone().detach() for k in range(10)],4).sum(axis=4).clone()
                            last_conv2_w_g_B[k] = (-1) * torch.stack([net[k].conv2.weight.grad.clone().detach() for k in range(10)],4).sum(axis=4).clone()
                            last_fc1_b_g_B[k] = (-1) * torch.stack([net[k].fc1.bias.grad.clone().detach() for k in range(10)],1).sum(axis=1).clone()
                            last_fc2_b_g_B[k] = (-1) * torch.stack([net[k].fc2.bias.grad.clone().detach() for k in range(10)],1).sum(axis=1).clone()
                            last_fc3_b_g_B[k] = (-1) * torch.stack([net[k].fc3.bias.grad.clone().detach() for k in range(10)],1).sum(axis=1).clone()
                            last_conv1_b_g_B[k] = (-1) * torch.stack([net[k].conv1.bias.grad.clone().detach() for k in range(10)],1).sum(axis=1).clone()
                            last_conv2_b_g_B[k] = (-1) * torch.stack([net[k].conv2.bias.grad.clone().detach() for k in range(10)],1).sum(axis=1).clone()
                            temp = []
                            temp.append(last_fc1_w_g_B[k].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_fc2_w_g_B[k].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_fc3_w_g_B[k].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_conv1_w_g_B[k].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_conv2_w_g_B[k].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_fc1_b_g_B[k].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_fc2_b_g_B[k].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_fc3_b_g_B[k].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_conv1_b_g_B[k].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_conv2_b_g_B[k].clone().view([-1]).unsqueeze(0).T)
                            para_cat[k] = torch.cat(temp,dim=0)
                    for k in range(Node-B):
                        if HW_sign[k+B]==1:#censoring
                            last_fc1_w_g_N[k+B] = net[k+B].fc1.weight.grad.clone()
                            last_fc2_w_g_N[k+B] = net[k+B].fc2.weight.grad.clone()
                            last_fc3_w_g_N[k+B] = net[k+B].fc3.weight.grad.clone()
                            last_conv1_w_g_N[k+B] = net[k+B].conv1.weight.grad.clone()
                            last_conv2_w_g_N[k+B] = net[k+B].conv2.weight.grad.clone()
                            last_fc1_b_g_N[k+B] = net[k+B].fc1.bias.grad.clone()
                            last_fc2_b_g_N[k+B] = net[k+B].fc2.bias.grad.clone()
                            last_fc3_b_g_N[k+B] = net[k+B].fc3.bias.grad.clone()
                            last_conv1_b_g_N[k+B] = net[k+B].conv1.bias.grad.clone()
                            last_conv2_b_g_N[k+B] = net[k+B].conv2.bias.grad.clone()
                            temp = []
                            temp.append(last_fc1_w_g_N[k+B].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_fc2_w_g_N[k+B].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_fc3_w_g_N[k+B].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_conv1_w_g_N[k+B].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_conv2_w_g_N[k+B].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_fc1_b_g_N[k+B].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_fc2_b_g_N[k+B].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_fc3_b_g_N[k+B].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_conv1_b_g_N[k+B].clone().view([-1]).unsqueeze(0).T)
                            temp.append(last_conv2_b_g_N[k+B].clone().view([-1]).unsqueeze(0).T)
                            para_cat[k+B] = torch.cat(temp,dim=0)
                    temp0 = []
                    temp0.append(last0_fc1_w_g.clone().view([-1]).unsqueeze(0).T)
                    temp0.append(last0_fc2_w_g.clone().view([-1]).unsqueeze(0).T)
                    temp0.append(last0_fc3_w_g.clone().view([-1]).unsqueeze(0).T)
                    temp0.append(last0_conv1_w_g.clone().view([-1]).unsqueeze(0).T)
                    temp0.append(last0_conv2_w_g.clone().view([-1]).unsqueeze(0).T)
                    temp0.append(last0_fc1_b_g.clone().view([-1]).unsqueeze(0).T)
                    temp0.append(last0_fc2_b_g.clone().view([-1]).unsqueeze(0).T)
                    temp0.append(last0_fc3_b_g.clone().view([-1]).unsqueeze(0).T)
                    temp0.append(last0_conv1_b_g.clone().view([-1]).unsqueeze(0).T)
                    temp0.append(last0_conv2_b_g.clone().view([-1]).unsqueeze(0).T)
                    para_0 = torch.cat(temp0,dim=0)

                    BucketGroup = [[] for g in range(BGroup)]
                    Node_list = list(range(Node))
                    random.shuffle(Node_list)
                    for g in range(BGroup):
                        BucketGroup[g] =  mean_aggregation([para_cat[Node_list[g*int(Node/BGroup)+ii]][None] for ii in range(int(Node/BGroup))])

                    TMLine = TMeans(BucketGroup, int(BGroup-2*B))

                    CMark = 0
                    net0.fc1.weight.grad = TMLine[CMark:CMark+fc1_w_len].T.view(fc1_w_size)
                    CMark += fc1_w_len
                    net0.fc2.weight.grad = TMLine[CMark:CMark+fc2_w_len].T.view(fc2_w_size)
                    CMark += fc2_w_len
                    net0.fc3.weight.grad = TMLine[CMark:CMark+fc3_w_len].T.view(fc3_w_size)
                    CMark += fc3_w_len
                    net0.conv1.weight.grad = TMLine[CMark:CMark+conv1_w_len].T.view(conv1_w_size)
                    CMark += conv1_w_len
                    net0.conv2.weight.grad = TMLine[CMark:CMark+conv2_w_len].T.view(conv2_w_size)
                    CMark += conv2_w_len
                    net0.fc1.bias.grad = TMLine[CMark:CMark+fc1_b_len].T.view(fc1_b_size)
                    CMark += fc1_b_len
                    net0.fc2.bias.grad = TMLine[CMark:CMark+fc2_b_len].T.view(fc2_b_size)
                    CMark += fc2_b_len
                    net0.fc3.bias.grad = TMLine[CMark:CMark+fc3_b_len].T.view(fc3_b_size)
                    CMark += fc3_b_len
                    net0.conv1.bias.grad = TMLine[CMark:CMark+conv1_b_len].T.view(conv1_b_size)
                    CMark += conv1_b_len
                    net0.conv2.bias.grad = TMLine[CMark:CMark+conv2_b_len].T.view(conv2_b_size)

                    if epoch==0:
                        HS_sign = [1]
                    else:
                        HS_sign = [1 if float(censor_S()) >= xi_s else 0]
                    HS_sum += sum(HS_sign)
                    optimizer0.step()

                    if HS_sign[0] == 1:  # censoring中心节点发送的
                        for k in range(Node):
                            last0_fc1_w_g = net0.fc1.weight.grad.clone()
                            last0_fc2_w_g = net0.fc2.weight.grad.clone()
                            last0_fc3_w_g = net0.fc3.weight.grad.clone()
                            last0_conv1_w_g = net0.conv1.weight.grad.clone()
                            last0_conv2_w_g = net0.conv2.weight.grad.clone()
                            last0_fc1_b_g = net0.fc1.bias.grad.clone()
                            last0_fc2_b_g = net0.fc2.bias.grad.clone()
                            last0_fc3_b_g = net0.fc3.bias.grad.clone()
                            last0_conv1_b_g = net0.conv1.bias.grad.clone()
                            last0_conv2_b_g = net0.conv2.bias.grad.clone()
                    for k in range(Node):
                        net[k].fc1.weight.grad = last0_fc1_w_g.clone()
                        net[k].fc2.weight.grad = last0_fc2_w_g.clone()
                        net[k].fc3.weight.grad = last0_fc3_w_g.clone()
                        net[k].conv1.weight.grad = last0_conv1_w_g.clone()
                        net[k].conv2.weight.grad = last0_conv2_w_g.clone()
                        net[k].fc1.bias.grad = last0_fc1_b_g.clone()
                        net[k].fc2.bias.grad = last0_fc2_b_g.clone()
                        net[k].fc3.bias.grad = last0_fc3_b_g.clone()
                        net[k].conv1.bias.grad = last0_conv1_b_g.clone()
                        net[k].conv2.bias.grad = last0_conv2_b_g.clone()
                    if epoch==0:
                        HW_sign = [1 for k in range(Node)]
                    else:
                        HW_sign = [1 if float(censor_W(k)) >= xi_w else 0 for k in range(Node)]
                    HW_sum += sum(HW_sign)
                    for k in range(Node):
                        optimizer[k].step()


                    net0_norm = ParaNorm1(net0)
                    if net0_norm>1000:
                        net0.fc1.weight.data = 1000*net0.fc1.weight.data.clone()/net0_norm
                        net0.fc2.weight.data = 1000*net0.fc2.weight.data.clone()/net0_norm
                        net0.fc3.weight.data = 1000*net0.fc3.weight.data.clone()/net0_norm
                        net0.conv1.weight.data = 1000*net0.conv1.weight.data.clone()/net0_norm
                        net0.conv2.weight.data = 1000*net0.conv2.weight.data.clone()/net0_norm
                        net0.fc1.bias.data = 1000*net0.fc1.bias.data.clone()/net0_norm
                        net0.fc2.bias.data = 1000*net0.fc2.bias.data.clone()/net0_norm
                        net0.fc3.bias.data = 1000*net0.fc3.bias.data.clone()/net0_norm
                        net0.conv1.bias.data = 1000*net0.conv1.bias.data.clone()/net0_norm
                        net0.conv2.bias.data = 1000*net0.conv2.bias.data.clone()/net0_norm
                    for k in range(Node):
                        netk_norm = ParaNorm1(net[k])
                        if netk_norm>1000:
                            net[k].fc1.weight.data = 1000*net[k].fc1.weight.data.clone()/netk_norm
                            net[k].fc2.weight.data = 1000*net[k].fc2.weight.data.clone()/netk_norm
                            net[k].fc3.weight.data = 1000*net[k].fc3.weight.data.clone()/netk_norm
                            net[k].conv1.weight.data = 1000*net[k].conv1.weight.data.clone()/netk_norm
                            net[k].conv2.weight.data = 1000*net[k].conv2.weight.data.clone()/netk_norm
                            net[k].fc1.bias.data = 1000*net[k].fc1.bias.data.clone()/netk_norm
                            net[k].fc2.bias.data = 1000*net[k].fc2.bias.data.clone()/netk_norm
                            net[k].fc3.bias.data = 1000*net[k].fc3.bias.data.clone()/netk_norm
                            net[k].conv1.bias.data = 1000*net[k].conv1.bias.data.clone()/netk_norm
                            net[k].conv2.bias.data = 1000*net[k].conv2.bias.data.clone()/netk_norm

                    if epoch%count_iter==0:
                        HS_sum_list.append(HS_sum)
                        HW_sum_list.append(HW_sum)
                        accur = Accuracy(net0, data_test, label_test)
                        accuracy_list.append(accur.cpu().numpy().tolist())
                        print(epoch,accuracy_list[int(epoch/count_iter)])
                        print(epoch,Accuracy(net[9], data_test, label_test).cpu().numpy().tolist())
                        print(epoch, HS_sum, HW_sum)
                        for k in range(Node):
                            outputs0[k] = net0(inputs[k])
                            loss0[k] = criterion(outputs0[k], labels[k])
                        running_loss += sum([loss0[k].item() for k in range(Node)])
                        loss_list.append(running_loss/(len(labels[0])*Node))

                print("----------finished training---------")
                pd.DataFrame(accuracy_list).to_csv('/home/hexchao/Simulation_CIFAR10/Acc_BTM240%d.csv'%pr_c)
                pd.DataFrame(HS_sum_list).to_csv('/home/hexchao/Simulation_CIFAR10/HSsum_BTM240%d.csv'%pr_c)
                pd.DataFrame(HW_sum_list).to_csv('/home/hexchao/Simulation_CIFAR10/HWsum_BTM240%d.csv'%pr_c)
                pd.DataFrame(loss_list).to_csv('/home/hexchao/Simulation_CIFAR10/Loss_BTM240%d.csv'%pr_c)
                pr_c += 1

