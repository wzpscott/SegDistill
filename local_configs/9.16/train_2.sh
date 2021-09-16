bash tools/dist_train.sh local_configs/9.16/attn_kl.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.16/attn+fea+logits_group.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.16/attn+fea+logits_kl.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.16/fea+logits_kl.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.16/logits_kl.py 8;
sleep 10;

