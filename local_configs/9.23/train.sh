bash tools/dist_train.sh local_configs/9.23/attn.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.23/logits1.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.23/logits2.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.23/baseline.py 8;
sleep 10;