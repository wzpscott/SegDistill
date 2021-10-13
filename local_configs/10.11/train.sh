bash tools/dist_train.sh local_configs/10.11/baseline.py 8;

bash tools/dist_train.sh local_configs/10.10/logits_c.py 8;
bash tools/dist_train.sh local_configs/10.10/logits_s.py 8;
bash tools/dist_train.sh local_configs/10.10/logits_c+s.py 8;

bash tools/dist_train.sh local_configs/10.10/logits_c_mask.py 8;
bash tools/dist_train.sh local_configs/10.10/logits_cg3.py 8;
bash tools/dist_train.sh local_configs/10.10/logits_sg8.py 8;

bash tools/dist_train.sh local_configs/10.10/attention_stage1.py 8;
bash tools/dist_train.sh local_configs/10.10/attention_stage2.py 8;
bash tools/dist_train.sh local_configs/10.10/attention_stage3.py 8;
bash tools/dist_train.sh local_configs/10.10/attention_stage4.py 8;
bash tools/dist_train.sh local_configs/10.10/attention.py 8;

bash tools/dist_train.sh local_configs/10.10/feature_stage1.py 8;
bash tools/dist_train.sh local_configs/10.10/feature_stage2.py 8;
bash tools/dist_train.sh local_configs/10.10/feature_stage3.py 8;
bash tools/dist_train.sh local_configs/10.10/feature_stage4.py 8;
bash tools/dist_train.sh local_configs/10.10/feature.py 8;