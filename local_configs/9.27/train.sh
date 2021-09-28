bash tools/dist_train.sh local_configs/9.27/logits_c.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.27/logits_s.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.27/logits_cg3.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.27/logits_cg6.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.27/logits_cg10.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/9.27/logits_sg.py 8;
sleep 10;

