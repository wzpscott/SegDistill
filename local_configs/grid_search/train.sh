bash tools/dist_train.sh local_configs/grid_search/baseline.py 2;
sleep 10s;
bash tools/dist_train.sh local_configs/grid_search/w=0.4.py 2;
sleep 10s;
# bash tools/dist_train.sh local_configs/grid_search/w=1.py 8;
# sleep 10s;
# bash tools/dist_train.sh local_configs/grid_search/w=5.py 8;
# sleep 10s;
# bash tools/dist_train.sh local_configs/grid_search/w=10.py 8;
# sleep 10s;