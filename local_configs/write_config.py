import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()
config_path = args.config_path

with open('/home/mist/SegformerDistillation/local_configs/train_10.21.sh','w') as f:
    for c in os.listdir(config_path):
        command = f'bash tools/dist_train.sh {config_path}{c} 8;\n'
        f.write(command)
print('done')