bash tools/dist_train.sh local_configs/10.24/MT_rot_1k.py 8;
bash tools/dist_train.sh local_configs/10.24/MT_rot_10k.py 8;
bash tools/dist_train.sh local_configs/10.24/MT_base.py 8;

# 新加的
bash tools/dist_train.sh local_configs/10.24/MT_rot_1.py 8;
bash tools/dist_train.sh local_configs/10.24/MT_rot_rand.py 8;
bash tools/dist_train.sh local_configs/10.24/b3_mask_bg.py 8;
bash tools/dist_train.sh local_configs/10.24/b3_mask_st.py 8;
bash tools/dist_train.sh local_configs/10.24/b3_mask_tm.py 8;
