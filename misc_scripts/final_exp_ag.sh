# datapath="G:/datasets/action_genome"
datapath="/mnt/DATA/datasets/action_genome"
project="../runs/sttran_gaze_ag"
epochs=25
warmup=3
save_period=-1

## wait for the best
# f0 gaze ca + global token
name=f0_final_ag
python train_ag.py --epochs $epochs --warmup $warmup --cfg configs/final/train_hyp_f0.yaml --data $datapath --project $project --name $name --gaze cross --global-token --save-period $save_period
python eval_inference_ag.py $project/$name/weights/last.pt --data $datapath --cfg configs/final/eval_hyp_f0.yaml --project $project --name $name --gaze cross --global-token
python eval_metrics_ag.py $project/$name/eval/all_results.json --data $datapath