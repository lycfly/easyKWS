import os
from config import Config
from dataset_prepare import *
from trainer import train_model,test_inference,finetune
from data import preprocess_mel, preprocess_mfcc, preprocess_wav, preprocess_mel_my
import torch
#from comvoice_data_proc import comvoice_data_proc
import warnings
warnings.filterwarnings("ignore")

list_fbank = [('mel', preprocess_mel_my)]
list_mfcc = [('mfcc', preprocess_mfcc)]
list_2d = [('mel', preprocess_mel), ('mfcc', preprocess_mfcc)]


def go(cfg_dict, preprocess_list, is_test, is_finetune):
    for p, preprocess_fun in preprocess_list:
        cfg = cfg_dict.copy()
        cfg['preprocess_fun'] = preprocess_fun
        cfg['CODER'] += '_%s' %p
        cfg['bagging_num'] = 1
        cfg['words'] = Config.words
        cfg['target_dataset'] = Config.dataset['name']
        if is_test:
            print("testing ", cfg['CODER'])
            test_inference(**cfg)
        elif is_finetune:
            print("finetuning ", cfg['CODER'])
            finetune(**cfg)
        else:
            print("training ", cfg['CODER'])
            train_model(**cfg)


'''
    Main function
'''
print('Easy KWS work Start !')
dataset_parpare(Config)
create_directroy(Config.model_path)
print('Dataset prepare Done !')

go(Config.model_config, list_fbank, Config.is_testing, False)

# finetuning
if Config.model_config['use_finetune']:
    go(Config.model_config, list_fbank, False, True)

    

#os.system('shutdown /s /t 0')