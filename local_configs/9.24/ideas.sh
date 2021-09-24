bash tools/dist_train.sh local_configs/9.24/sa_ca_kl.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.24/logits_group.py 8;
sleep 10;