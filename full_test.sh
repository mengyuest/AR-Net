if [ "$#" -eq 2 ]
then
    DATA_DIR=$1
    MODEL_DIR=$2
else
    DATA_DIR=../../datasets/activity-net-v1.3
    MODEL_DIR=../../logs_tsm/test_suite/ar-net
fi

echo "Using data path: ${DATA_DIR} and model path: ${MODEL_DIR}"


#TODO 0. Test for Adaptive ResNet (~73.8)
python -u main_base.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --accuracy_weight 0.9 --efficency_weight 0.1 --exp_decay --init_tau 0.000001 --policy_also_backbone --policy_input_offset 3 --uniform_loss_weight 3.0 --use_gflops_loss --batch-size 48 -j 16 --gpus 0 1 2 3 --test_from ${MODEL_DIR}/exp0_res_ada.pth.tar --data_dir ${DATA_DIR} | tee tmp_log.txt
OUTPUT0=`cat tmp_log.txt | tail -n 3`

#TODO 1. Test for SCSampler (~72.9)
python -u main_base.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --backbone_list resnet50  --reso_list 224 84 --policy_input_offset 0 --frame_independent --real_scsampler --consensus_type scsampler --batch-size 48 -j 16  --gpus 0 1 2 3 --test_from ${MODEL_DIR}/exp1_res_scsampler.pth.tar --data_dir ${DATA_DIR} | tee tmp_log.txt
OUTPUT1=`cat tmp_log.txt | tail -n 3`

#TODO 2. Test for Baseline TSN (~72.5)
python -u main_base.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone resnet50 --offline_lstm_all  --frame_independent --batch-size 48 -j 16 --gpus 0 1 2 3 --test_from ${MODEL_DIR}/exp2_res_uniform.pth.tar --data_dir ${DATA_DIR} | tee tmp_log.txt
OUTPUT2=`cat tmp_log.txt | tail -n 3`

#TODO 3. Test for Baseline LSTM (~71.2)
python -u main_base.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone resnet50 --backbone_list resnet50 --offline_lstm_all --batch-size 48 -j 16  --gpus 0 1 2 3 --test_from ${MODEL_DIR}/exp3_res_lstm.pth.tar --data_dir ${DATA_DIR} | tee tmp_log.txt
OUTPUT3=`cat tmp_log.txt | tail -n 3`

#TODO 4. Test for Adaptive RANDOM (~65.7)
python -u main_base.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --policy_also_backbone --policy_input_offset 3 --use_gflops_loss --random_policy  --batch-size 48 -j 16 --gpus 0 1 2 3 --test_from ${MODEL_DIR}/exp4_res_rand.pth.tar --data_dir ${DATA_DIR} | tee tmp_log.txt
OUTPUT4=`cat tmp_log.txt | tail -n 3`

#TODO 5. Test for Adaptive ALL (~73.5)
python -u main_base.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --reso_list 224 168 112 84 --skip_list 1 --backbone_list resnet50 resnet34 resnet18 --all_policy --policy_also_backbone --policy_input_offset 3 --batch-size 48 -j 16  --gpus 0 1 2 3 --test_from ${MODEL_DIR}/exp5_res_all.pth.tar --data_dir ${DATA_DIR} | tee tmp_log.txt
OUTPUT5=`cat tmp_log.txt | tail -n 3`

#TODO 6. Test for Adaptive EfficientNet (~79.7)
python -u main_base.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list efficientnet-b3 efficientnet-b1 efficientnet-b0 --skip_list 1 2 4 --accuracy_weight 0.9 --efficency_weight 0.1 --policy_also_backbone --policy_input_offset 3 --uniform_loss_weight 5.0 --use_gflops_loss --init_tau 0.000001 --batch-size 48 -j 16  --gpus 0 1 2 3 --test_from ${MODEL_DIR}/exp6_eff_ada.pth.tar --data_dir ${DATA_DIR} | tee tmp_log.txt
OUTPUT6=`cat tmp_log.txt | tail -n 3`

#TODO 7. New S.O.T.A. Baseline TSN using updated training logics (~75.6)
python -u main_base.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --batch-size 48  -j 36 --gpus 0 1 2 3 --test_from ${MODEL_DIR}/exp7_res_uni_new.pth.tar --data_dir ${DATA_DIR} | tee tmp_log.txt

OUTPUT7=`cat tmp_log.txt | tail -n 3`

#TODO 8. New S.O.T.A. AR-Net(resnet) using updated training logics (~76.8)
python -u main_base.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 --backbone_list resnet50 resnet34 resnet18 --skip_list 1 2 4 --accuracy_weight 0.9 --efficency_weight 0.1 --exp_decay --init_tau 0.000001 --policy_also_backbone --policy_input_offset 3 --uniform_loss_weight 3.0 --use_gflops_loss --batch-size 48 -j 16 --gpus 0 1 2 3 --test_from ${MODEL_DIR}/exp8_res_ada_new.pth.tar --data_dir ${DATA_DIR} | tee tmp_log.txt
OUTPUT8=`cat tmp_log.txt | tail -n 3`

echo -e "\n\033[1;36mEXPECT   : 73.830 (exp0_res_ada)\033[0m"
echo $OUTPUT0
echo -e "\n\033[1;36mEXPECT   : 72.937 (exp1_res_scsampler)\033[0m"
echo $OUTPUT1
echo -e "\n\033[1;36mEXPECT   : 72.495 (exp2_res_uniform)\033[0m"
echo $OUTPUT2
echo -e "\n\033[1;36mEXPECT   : 71.176 (exp3_res_lstm)\033[0m"
echo $OUTPUT3
echo -e "\n\033[1;36mEXPECT   : 65.657 (exp4_res_rand)\033[0m"
echo $OUTPUT4
echo -e "\n\033[1;36mEXPECT   : 73.454 (exp5_res_all)\033[0m"
echo $OUTPUT5
echo -e "\n\033[1;36mEXPECT   : 79.697 (exp6_eff_ada)\033[0m"
echo $OUTPUT6
echo -e "\n\033[1;36mEXPECT   : 75.589 (exp7_res_uni_new)\033[0m"
echo $OUTPUT7
echo -e "\n\033[1;36mEXPECT   : 76.791 (exp8_res_ada_new)\033[0m"
echo $OUTPUT8