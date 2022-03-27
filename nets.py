from basic_nets import *
from bc_resnet import *
from Dscnn import *

from torchsummary import summary

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model_info = [
        {'name':'resnet', 'net':ResModel},            #0
        {'name':'bc-resnet', 'net':BCResNet},         #1
        {'name':'gru', 'net':GRU_Model},              #2
        {'name':'dscnn', 'net':DSCNN},                #3
        {'name':'my-bc-resnet', 'net':myBCResNet},    #4
        {'name':'qdscnn', 'net':qDSCNN},              #5
        {'name':'q-mybc-resnet', 'net':myQBCResNet},  #6

    ]
    model = model_info[6]
    net = model['net'](label_num=4).to(device)
    summary(net, input_size = (1,40,61))