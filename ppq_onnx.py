'''
@Time    : 2022.04.23
@Author  : wuruidong
@Email   : wuruidong@hotmail.com
@FileName: ppq_onnx.py 源文件参考PPQ中ProgramEntrance.py脚本
@Software: python pytorch=1.6.0 ppq=0.6.3 onnx=1.8.1
@Cnblogs : https://www.cnblogs.com/ruidongwu
'''
from ppq import *
from ppq.api import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# modify configuration below:
WORKING_DIRECTORY = 'working'                             # choose your working directory
TARGET_PLATFORM   = TargetPlatform.NXP_INT8          # choose your target platform
MODEL_TYPE        = NetworkFramework.ONNX                 # or NetworkFramework.CAFFE
INPUT_LAYOUT          = 'chw'                             # input data layout, chw or hwc
NETWORK_INPUTSHAPE    = [16, 1, 28, 28]                  # input shape of your network
CALIBRATION_BATCHSIZE = 16                                # batchsize of calibration dataset
EXECUTING_DEVICE      = 'cuda'                            # 'cuda' or 'cpu'.
REQUIRE_ANALYSE       = True
DUMP_RESULT           = False

# -------------------------------------------------------------------
# SETTING 对象用于控制 PPQ 的量化逻辑
# 当你的网络量化误差过高时，你需要修改 SETTING 对象中的参数来进行特定的优化
# -------------------------------------------------------------------
SETTING = UnbelievableUserFriendlyQuantizationSetting(
    platform = TARGET_PLATFORM, finetune_steps = 2500,
    finetune_lr = 1e-3, calibration = 'percentile',
    equalization = True, non_quantable_op = None)
SETTING = SETTING.convert_to_daddy_setting()

print('正准备量化你的网络，检查下列设置:')
print(f'WORKING DIRECTORY    : {WORKING_DIRECTORY}')
print(f'TARGET PLATFORM      : {TARGET_PLATFORM.name}')
print(f'NETWORK INPUTSHAPE   : {NETWORK_INPUTSHAPE}')
print(f'CALIBRATION BATCHSIZE: {CALIBRATION_BATCHSIZE}')


mnist = datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor()]))
mnist_data = mnist.data.view(-1, 1, 28, 28).float()
dataset_len = mnist_data.shape[0]
#mnist_data = mnist_data/255
calibration_dataset = mnist_data

dataloader = DataLoader(
    dataset=calibration_dataset,
    batch_size=32, shuffle=True)

print('网络正量化中，根据你的量化配置，这将需要一段时间:')
quantized = quantize(
    working_directory=WORKING_DIRECTORY, setting=SETTING,
    model_type=MODEL_TYPE, executing_device=EXECUTING_DEVICE,
    input_shape=NETWORK_INPUTSHAPE, target_platform=TARGET_PLATFORM,
    dataloader=dataloader, calib_steps=256)

# -------------------------------------------------------------------
# 如果你需要执行量化后的神经网络并得到结果，则需要创建一个 executor
# 这个 executor 的行为和 torch.Module 是类似的，你可以利用这个东西来获取执行结果
# 请注意，必须在 export 之前执行此操作。
# -------------------------------------------------------------------
executor = TorchExecutor(graph=quantized)
# output = executor.forward(input)

# -------------------------------------------------------------------
# 导出 PPQ 执行网络的所有中间结果，该功能是为了和硬件对比结果
# 中间结果可能十分庞大，因此 PPQ 将使用线性同余发射器从执行结果中采样
# 为了正确对比中间结果，硬件执行结果也必须使用同样的随机数种子采样
# 查阅 ppq.util.fetch 中的相关代码以进一步了解此内容
# 查阅 ppq.api.fsys 中的 dump_internal_results 函数以确定采样逻辑
# -------------------------------------------------------------------
if DUMP_RESULT:
    dump_internal_results(
        graph=quantized, dataloader=dataloader,
        dump_dir=WORKING_DIRECTORY, executing_device=EXECUTING_DEVICE)

# -------------------------------------------------------------------
# PPQ 计算量化误差时，使用信噪比的倒数作为指标，即噪声能量 / 信号能量
# 量化误差 0.1 表示在整体信号中，量化噪声的能量约为 10%
# 你应当注意，在 graphwise_error_analyse 分析中，我们衡量的是累计误差
# 网络的最后一层往往都具有较大的累计误差，这些误差是其前面的所有层所共同造成的
# 你需要使用 layerwise_error_analyse 逐层分析误差的来源
# -------------------------------------------------------------------
print('正计算网络量化误差(SNR)，最后一层的误差应小于 0.1 以保证量化精度:')
reports = graphwise_error_analyse(
    graph=quantized, running_device=EXECUTING_DEVICE, steps=256,
    dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
for op, snr in reports.items():
    if snr > 0.1: ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

if REQUIRE_ANALYSE:
    print('正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:')
    layerwise_error_analyse(graph=quantized, running_device=EXECUTING_DEVICE, steps=256,
                            interested_outputs=None,
                            dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

print('网络量化结束，正在生成目标文件:')
export(working_directory=WORKING_DIRECTORY,
       quantized=quantized, platform=TargetPlatform.ONNXRUNTIME)
       #使用NXP_INT8导出浮点权重表示方法的模型，使用ONNXRUNTIME导出带有原始整形权重表示方法的模型

# 如果你需要导出 CAFFE 模型，使用下面的语句
#export(working_directory=WORKING_DIRECTORY,
#       quantized=quantized, platform=TARGET_PLATFORM,
#       input_shapes=[NETWORK_INPUTSHAPE])