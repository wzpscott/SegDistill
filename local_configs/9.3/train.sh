bash tools/dist_train.sh local_configs/9.3/logits+attn+mlp.py 8;
sleep 10s;
bash tools/dist_train.sh local_configs/9.3/attn.py 8;
sleep 10s;
bash tools/dist_train.sh local_configs/9.3/logits1.py 8;
sleep 10s;
bash tools/dist_train.sh local_configs/9.3/logits2.py 8;
sleep 10s;