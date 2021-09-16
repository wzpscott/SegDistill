bash tools/dist_train.sh local_configs/9.16/b0_baseline.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.16/attn_ca.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.16/attn+fea+logits_ca.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.16/fea+logits_ca.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.16/logits_ca.py 8;
sleep 10;

