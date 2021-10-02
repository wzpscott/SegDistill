bash tools/dist_train.sh local_configs/10.2/attention_cg4.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/10.2/attention_cg8.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/10.2/attention_cg16.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/10.2/attention_sg[8,4,2,1].py 8;
sleep 10;
bash tools/dist_train.sh local_configs/10.2/attention_sg4.py 8;
sleep 10;

bash tools/dist_train.sh local_configs/10.2/feature_cg4.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/10.2/feature_cg8.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/10.2/feature_sg4.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/10.2/feature_sg[8,4,2,1].py 8;
sleep 10;

bash tools/dist_train.sh local_configs/10.2/logits_cg3_mask.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/10.2/logits_sg16_8.py 8;
sleep 10;
bash tools/dist_train.sh local_configs/10.2/logits_sg16.py 8;
sleep 10;