import os
import json

def mkcityjson(command):
    name = command.split(" ")[-2].split('/')[-1].split(".")[0]
    city = {
        "Token":"Tm6ehGHVRfwY9ssjdPAc5A",
        "business_flag": "youtu_lowsource_chongqing",
        "host_num": 1,
        "host_gpu_num": 8,
        "image_full_name": "mirrors.tencent.com/rpf/pytorch:1.7.0",
        "model_local_file_path": "/mnt/private/SegformerDistillation/",
        "task_flag": "",
        "readable_name": name,
        "GPUName": "V100",
        "is_elasticity": True,
        "start_cmd": command,
        "init_cmd": "pip3 install -r requirements.txt && pip3 install -e . && cp -r /apdcephfs/private_inchzhang/pretrained /dev/shm/ && ln -s /dev/shm/pretrained && mkdir /dev/shm/cityscapes && cp -r /youtu-public/cityscapes/leftImg8bit /dev/shm/cityscapes/ &&  cp -r /youtu-public/cityscapes/gtFine /dev/shm/cityscapes/ && ln -s  /dev/shm/cityscapes data/cityscapes && python3 tools/convert_datasets/cityscapes.py data/cityscapes",
        "exit_cmd": "",
        "enable_evicted_pulled_up": True,
        "enable_evicted_end_task": True,
        "cuda_version": "10.1",
        "dataset_id": "FC5EA28881E4406CB9FFC6EB988154FD"
    }
    return city


lines = open('local_configs/train_10.25.sh','r').read().splitlines()
print(lines)
for i in lines:
    f = open("/mnt/private/SegformerDistillation/start.json",'w')
    c = mkcityjson(i)
    f.write(json.dumps(c,ensure_ascii=False))
    f.close()
    cmd = "jizhi_client start -scfg start.json"
    os.system(cmd)
