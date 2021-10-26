import os
import json

def mkadejson(command):
    name = command.split(" ")[-2].split('/')[-1].split(".")[0]
    ade = {
        "Token": "Tm6ehGHVRfwY9ssjdPAc5A",
        "business_flag": "youtu_lowsource_chongqing",
        "host_num": 1,
        "host_gpu_num": 8,
        "image_full_name": "mirrors.tencent.com/rpf/pytorch:1.7.0",
        "model_local_file_path": "/mnt/private/SegformerDistillation/",
        "keep_alive": True,
        "task_flag": "",
        "readable_name": name,
        "GPUName": "V100",
        "is_elasticity": True,
        "start_cmd": command,
        "init_cmd": "pip3 install -r requirements.txt && pip3 install -e . && cp -r /apdcephfs/private_inchzhang/pretrained /dev/shm/ && ln -s /dev/shm/pretrained && cp -r /youtu-reid/zhangyinqi/dataset/ade /dev/shm/ && ln -s  /dev/shm/ade data/ade",
        "exit_cmd": "",
        "enable_evicted_pulled_up": True,
        "enable_evicted_end_task": True,
        "cuda_version": "10.1",
        "dataset_id": "1A777126FC8049378C4C2B46FE67158E"
    }
    return ade


lines = open('/mnt/private/SegformerDistillation/local_configs/train_10.26.sh','r').read().splitlines()
print(lines)
for i in lines:
    f = open("/mnt/private/SegformerDistillation/start.json",'w')
    c = mkadejson(i)
    f.write(json.dumps(c,ensure_ascii=False))
    f.close()
    cmd = "jizhi_client start -scfg start.json"
    os.system(cmd)
