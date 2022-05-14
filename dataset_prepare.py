import numpy as np
import shutil
import argparse
import os
from config import Config
    
def create_directroy(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def move_gsdc_files(src_folder, to_folder, list_file):
    with open(list_file) as f:
        for line in f.readlines():
            line = line.rstrip()
            dirname = os.path.dirname(line)
            dest = os.path.join(to_folder, dirname)
            if not os.path.exists(dest):
                os.mkdir(dest)
            shutil.move(os.path.join(src_folder, line), dest)


def download_GSC(config):
    version = config.dataset['version']
    dataset_name = 'google_speech_commands'
    dataset_path = os.path.join(config.data_dir,dataset_name)
    tar_path = os.path.join(dataset_path, 'speech_commands_v0.0{}'.format(version))
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    # download data
    if os.path.exists(tar_path):
        print('Skipping download GSC dataset ...')
    elif os.path.exists(os.path.join(dataset_path, 'speech_commands_v0.0{}.tar.gz'.format(version))):
        print('Skipping download GSC dataset ...')
    else:    
        print('Downloading GSC dataset ...')
        os.system(f'wget -P {dataset_path} '+ 'http://download.tensorflow.org/data/speech_commands_v0.0%s.tar.gz' % (version))
    
    if not os.path.exists(tar_path): 
        os.makedirs(tar_path)      
        print('Extracing GSC dataset ...')
        os.system(f'tar -xf {os.path.join(dataset_path, "speech_commands_v0.0%s.tar.gz" %(version))} -C {tar_path}')
    else:
        print('Skipping extract GSC dataset ...')

def prepare_GSDC(config):
    dataname = config.dataset['name']
    version = config.dataset['version']
    raw_dir = 'speech_commands_v0.0{}'.format(version)
    root_path = os.path.join(config.data_dir, 'google_speech_commands')
    audio_folder = os.path.join(root_path, raw_dir)
    work_folder  = os.path.join(root_path,dataname)
    if not os.path.exists(work_folder):
        validation_path = os.path.join(audio_folder, 'validation_list.txt')
        test_path = os.path.join(audio_folder, 'testing_list.txt')

        create_directroy(work_folder)
        valid_folder = os.path.join(work_folder, 'valid')
        test_folder = os.path.join(work_folder, 'test')
        train_folder = os.path.join(work_folder, 'train')
        create_directroy(valid_folder)
        create_directroy(test_folder)
        create_directroy(train_folder)
        print("valid folder name: " + valid_folder)
        print("test folder name: " + test_folder)
        print("train folder name: " + train_folder)

        shutil.copytree(audio_folder, train_folder, dirs_exist_ok=True)
        move_gsdc_files(train_folder, test_folder, test_path)
        move_gsdc_files(train_folder, valid_folder, validation_path)
    print("*** GSDC Dataset Allready prepared ! ***")



def dataset_parpare(config):
    if config.dataset['name'] == 'GSDC_v01' or config.dataset['name'] == 'GSDC_v02':
        download_GSC(config)
        prepare_GSDC(config)
        
if __name__ == '__main__':
    download_GSC(Config)
    prepare_GSDC(Config)