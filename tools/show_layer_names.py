import argparse
from mmseg.models import build_segmentor
from mmcv.utils import Config

parser = argparse.ArgumentParser(description='show layer names')
parser.add_argument('config', help='train config file path')
args = parser.parse_args()
cfg = Config.fromfile(args.config).model
cfg_s = cfg['cfg_s']
cfg_t = cfg['cfg_t']
train_cfg = cfg['train_cfg']
test_cfg = cfg['test_cfg']


student = build_segmentor(
    cfg_s, train_cfg=train_cfg, test_cfg=test_cfg)

teacher = build_segmentor(
    cfg_t, train_cfg=train_cfg, test_cfg=test_cfg)

print('Teacher Layers')
for name, module in teacher.named_modules():
    print(name)
print('----------------------------------------')
print('Student Layers')
for name, module in student.named_modules():
    print(name)