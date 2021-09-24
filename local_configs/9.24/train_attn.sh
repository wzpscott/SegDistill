bash tools/dist_train.sh local_configs/9.24/attn_w=2.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.24/attn_w=4.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.24/attn_w=8.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.24/attn_w=16.py 8;
sleep 10;