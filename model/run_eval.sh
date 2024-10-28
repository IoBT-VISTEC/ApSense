# declare -a arr=("MiniRRWaveNet" "SVC" "XGB" "RF")
# declare -a arr=("MiniRRWaveNet" "EEGNetSA" "AIOSA" "DSepNetSmall")
declare -a arr=("DSepST15Net")
# declare -a arr=(20)
# declare -a arr=("MiniRRWaveNet" "DSepNet" "XGB" "RF")

# features="pa"

## now loop through the above array
for i in "${arr[@]}"
do
#    python evaluate.py --dataset mesa --model "$i" --dataset_dir /mount/guntitats/apsens_processed --gpu 1
#    python evaluate.py --dataset heartbeat --model "$i" --dataset_dir /mount/guntitats/apsens_processed --gpu 1
   
#    python evaluate.py --dataset heartbeat --model "$i" --dataset_dir /mount/guntitats/apsens_processed --log_dir baseline --weight_dir baseline --gpu 0,2
   
   python evaluate.py --dataset heartbeat --model "$i" --dataset_dir /mount/guntitats/apsens_processed_aug --log_dir enhanced --weight_dir enhanced --gpu 2
   
#    python evaluate.py --dataset mesa --model "$i" --dataset_dir /mount/guntitats/apsens_processed_aug --log_dir features/$features --weight_dir features/$features --features $features --gpu 3
   
#    python evaluate.py --dataset heartbeat --model "$i" --dataset_dir /mount/guntitats/apsens_processed_ns --log_dir ns --weight_dir ns --gpu 0,2
    
#     python evaluate_regression.py --dataset mesa --model "$i" --save_pred /mount/guntitats/apsens/regr --dataset_dir /mount/guntitats/apsens_processed_10m --log_dir regr --weight_dir regr --gpu 2

#     python evaluate_finetune.py --finetune_size $i --dataset mesa --model MiniRRWaveNet --dataset_dir /mount/guntitats/apsens_processed_aug --log_dir enhanced --weight_dir enhanced --gpu 1
    

#     python evaluate_finetune.py --finetune_size $i --dataset mesa --model DSepNet --dataset_dir /mount/guntitats/apsens_processed_aug --log_dir enhanced --weight_dir enhanced --gpu 1


#     python evaluate_finetune.py --finetune_size $i --dataset heartbeat --model MiniRRWaveNet --dataset_dir /mount/guntitats/apsens_processed_aug --log_dir enhanced --weight_dir enhanced --gpu 1
    

#     python evaluate_finetune.py --finetune_size $i --dataset heartbeat --model DSepNet --dataset_dir /mount/guntitats/apsens_processed_aug --log_dir enhanced --weight_dir enhanced --gpu 1,2
done