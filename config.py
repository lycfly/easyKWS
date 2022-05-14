import os
from nets import *

class Config(object):
    data_dir   = '/root'  # autodl 
    #data_dir   = '../autodl-tmp/datasets'
    model_path = '../autodl-tmp/model'
    log_path =   '../tf-logs'
    dataset_name = 'GSDC_v01'
    words = ["wow", "marvin"]
    is_testing = False

    dataset_dict = {
        'GSDC_v01' : {'name':'GSDC_v01', 'version':1, "path": os.path.join(data_dir, 'google_speech_commands/GSDC_v01')},   #0
        'GSDC_v02' : {'name':'GSDC_v02', 'version':2, "path": os.path.join(data_dir, 'google_speech_commands/GSDC_v02')},   #1
        'ComVoice' : {'name':'ComVoice', 'version':0},   #3
    }
    noise_path = {
        "GSDC" : os.path.join(data_dir, 'google_speech_commands/GSDC_v01/train/_background_noise_') ,
        "other" : None
    }
    dataset  = dataset_dict[dataset_name]
    noise_path = noise_path['GSDC']             # path cotain noise wavs
    

    model_id   = 4
    model_info = [
        {'name':'resnet', 'net':ResModel},            #0
        {'name':'bc-resnet', 'net':BCResNet},         #1
        {'name':'gru', 'net':GRU_Model},              #2
        {'name':'dscnn', 'net':DSCNN},                #3
        {'name':'my-bc-resnet', 'net':myBCResNet},    #4
        {'name':'qdscnn', 'net':qDSCNN},              #5
        {'name':'q-mybc-resnet', 'net':myQBCResNet},  #6

    ]

    model = model_info[model_id]

    model_config = {

    'use_finetune'    :   False,
    'epochs_ft'       :   5, 
    'learning_rate_ft':   0.0001,

    'log_path'   :   log_path,
    'model_path' :   model_path,
    'data_path'  :   dataset['path'],
    'noise_path' :   noise_path,
    'num_workers':   128,
    'model_class':   model['net'],
    'CODER'      :   model['name'],

    'unknown_ratio'   :   0.2,

    'epochs'          :   50, #250
    'BATCH_SIZE'      :   128,
    'learning_rate'   :   0.1,
    'warmup'          :   True,

    'is_1d'           :   False,
    'reshape_size'    :   None,



    'send_msg'        :   True,

    'feature_cfg'     :  {
        'feature_use_gpu' : False,
        'use_quantize'    : True,
        'n_fft'           : 256,
        'sr'              : 16000,
        'win_length'      : 256,
        'hop_length'      : 256,
        'n_mels'          : 40,
        'fmin'            : 20,
        'fmax'            : 4000,

        'qlist'           :{
            'win'         : [0, 8, 7],
            'fft'         : [1,20,15],
            'pow'         : [0,20,15],
            'mel'         : [0, 4, 4],
            'spec'        : [0,20,15],
            'log'         : [1, 8, 1],
            'logtan'      : [0, 8, 1],

        }
    }
}