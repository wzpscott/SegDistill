bash tools/dist_train.sh local_configs/9.1/logits+attn+mlp.py 8;
sleep 10s;
bash tools/dist_train.sh local_configs/9.1/attn.py 8;
sleep 10s;
bash tools/dist_train.sh local_configs/9.1/logits1.py 8;
sleep 10s;