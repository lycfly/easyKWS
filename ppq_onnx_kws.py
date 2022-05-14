
from ppq import *
from ppq.api import *
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data import get_label_dict, get_wav_list, SpeechDataset, preprocess_mel, preprocess_mfcc, preprocess_wav, preprocess_mel_my
from config import Config
import warnings
warnings.filterwarnings("ignore")

# pth to onnx
CODER = Config.model_config['CODER']
pth_name = os.path.join(Config.model_path,"model_best_acc_%s_mel.pth" %(CODER))
net = Config.model['net'](label_num=len(Config.words)+2, finetune=False)
net.load_state_dict(torch.load(pth_name))
net.eval().cuda()

input1 = torch.randn(16, 1, 40, 61).cuda()
input_names = [ "input"]
output_names = [ "output" ]
torch.onnx.export(net, input1, os.path.join(Config.model_path,"model.onnx"), verbose=True, input_names=input_names, output_names=output_names)
print('Pth to onnx finish!')

# modify configuration below:
WORKING_DIRECTORY = Config.model_path                 # choose your working directory
TARGET_PLATFORM   = TargetPlatform.NXP_INT8          # choose your target platform
MODEL_TYPE        = NetworkFramework.ONNX                 # or NetworkFramework.CAFFE
INPUT_LAYOUT          = 'chw'                             # input data layout, chw or hwc
NETWORK_INPUTSHAPE    = [16, 1, 40, 61]                  # input shape of your network
CALIBRATION_BATCHSIZE = 16                                # batchsize of calibration dataset
EXECUTING_DEVICE      = 'cuda'                            # 'cuda' or 'cpu'.
REQUIRE_ANALYSE       = True
DUMP_RESULT           = False

# -------------------------------------------------------------------
# SETTING 对象用于控制 PPQ 的量化逻辑
# 当你的网络量化误差过高时，你需要修改 SETTING 对象中的参数来进行特定的优化
# -------------------------------------------------------------------
SETTING = UnbelievableUserFriendlyQuantizationSetting(
    platform = TARGET_PLATFORM, finetune_steps = 100,
    finetune_lr = 1e-3, calibration = 'percentile',
    equalization = True, non_quantable_op = None)
SETTING = SETTING.convert_to_daddy_setting()

print('正准备量化你的网络，检查下列设置:')
print(f'WORKING DIRECTORY    : {WORKING_DIRECTORY}')
print(f'TARGET PLATFORM      : {TARGET_PLATFORM.name}')
print(f'NETWORK INPUTSHAPE   : {NETWORK_INPUTSHAPE}')
print(f'CALIBRATION BATCHSIZE: {CALIBRATION_BATCHSIZE}')


data_path = Config.model_config['data_path']
unknown_ratio = Config.model_config['unknown_ratio']
noise_path = Config.noise_path
target_dataset = Config.dataset['name']
preprocess_fun = preprocess_mel_my
feature_cfg = Config.model_config['feature_cfg']
reshape_size =  Config.model_config['reshape_size']
is_1d = Config.model_config['is_1d']

label_to_int, int_to_label = get_label_dict(Config.words)
_, _, test_list = get_wav_list(path=data_path, target_dataset=target_dataset, words=label_to_int.keys(),
                                                unknown_ratio=unknown_ratio)
testdataset  = SpeechDataset(noise_path=noise_path, mode='test', label_words_dict=label_to_int,
                                wav_list=test_list[:1600], add_noise=False, preprocess_fun=preprocess_fun,
                                feature_cfg=feature_cfg, resize_shape=reshape_size,is_1d=is_1d,ppq_enable=True)
calibration_dataset = testdataset
dataloader = DataLoader(
    dataset=calibration_dataset,
    batch_size=16, shuffle=True)

print('网络正量化中，根据你的量化配置，这将需要一段时间:')
quantized = quantize(
    working_directory=WORKING_DIRECTORY, setting=SETTING,
    model_type=MODEL_TYPE, executing_device=EXECUTING_DEVICE,
    input_shape=NETWORK_INPUTSHAPE, target_platform=TARGET_PLATFORM,
    dataloader=dataloader, calib_steps=100)

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