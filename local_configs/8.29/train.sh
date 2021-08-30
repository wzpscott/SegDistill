bash tools/dist_train.sh local_configs/8.29/1.py 2;
sleep 10s;
bash tools/dist_train.sh local_configs/8.29/distill.py 2;
sleep 10s;
bash tools/dist_train.sh local_configs/8.29/2.py 2;
sleep 10s;
bash tools/dist_train.sh local_configs/8.29/distill_T=2.py 2;
sleep 10s;