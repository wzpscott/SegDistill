bash tools/dist_train.sh local_configs/9.20/attn.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.20/baseline.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.20/logits1.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.20/logits2.py 8;
sleep 10;