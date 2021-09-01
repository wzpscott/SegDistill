bash tools/dist_train.sh local_configs/9.1/logits+attn+mlp.py 2;
sleep 10s;
bash tools/dist_train.sh local_configs/9.1/attn.py 2;
sleep 10s;