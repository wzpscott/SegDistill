import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('sh_path',default='train_10.24.sh')
args = parser.parse_args()
config_path = args.config_path
sh_path = args.sh_path

with open(f'/home/mist/SegformerDistillation/local_configs/{sh_path}','w') as f:
    for c in os.listdir(config_path):
        if 'example' not in c:
            command = f'bash tools/dist_train.sh {config_path}{c} 8;\n'
            f.write(command)
print('done')

