# TinySalientSleepNet_taiyang_green

python -u train_shhs_unet.py --gpu_device 3 --dataset ISRUC3 --ISRUC3_channel C4_A1,ROC_A1 --input_epoch_num 120 --num_epochs 1 --save_dir result/test_1epoch/ |& tee logs/test_1epoch.log

# 0317确定最优Tiny模型
python -u train_shhs_unet.py --gpu_device 3 --input_epoch_num 120 --testonly true --dataset taiyang --taiyang_channel 'EEG C4-REF,EOG1' --save_dir result/IIMS_len120/ | tee logs/best0317_taiyang_IIMS_len120.log

