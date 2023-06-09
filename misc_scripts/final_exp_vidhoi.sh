# datapath="G:/datasets/VidOR"
datapath="/mnt/DATA/datasets/VidOR/"
project="../runs/sttran_gaze_vidhoi_final"
epochs=25
warmup=3
save_period=-1


## wait for the best
# f0 gaze ca + global token
name=f0_final
python train_vidhoi.py --epochs $epochs --warmup $warmup --cfg configs/final/train_hyp_f0.yaml --data $datapath --project $project --name $name --gaze cross --global-token --save-period $save_period
python eval_inference_vidhoi.py $project/$name/weights/last.pt --data $datapath --cfg configs/ablation_final/eval_hyp_f0.yaml --project $project --name $name --gaze cross --global-token
python eval_metrics_vidhoi.py $project/$name/eval/all_results.json --data $datapath

# f1 gaze ca + global token
name=f1_final
python train_vidhoi.py --epochs $epochs --warmup $warmup --cfg configs/final/train_hyp_f1.yaml --data $datapath --project $project --name $name --gaze cross --global-token --save-period $save_period
python eval_inference_vidhoi.py $project/$name/weights/last.pt --data $datapath --cfg configs/ablation_final/eval_hyp_f1.yaml --project $project --name $name --gaze cross --global-token
python eval_metrics_vidhoi.py $project/$name/eval/all_results.json --data $datapath

# f3 gaze ca + global token
name=f3_final
python train_vidhoi.py --epochs $epochs --warmup $warmup --cfg configs/final/train_hyp_f3.yaml --data $datapath --project $project --name $name --gaze cross --global-token --save-period $save_period
python eval_inference_vidhoi.py $project/$name/weights/last.pt --data $datapath --cfg configs/ablation_final/eval_hyp_f3.yaml --project $project --name $name --gaze cross --global-token
python eval_metrics_vidhoi.py $project/$name/eval/all_results.json --data $datapath

# f5 gaze ca + global token
name=f5_final
python train_vidhoi.py --epochs $epochs --warmup $warmup --cfg configs/final/train_hyp_f5.yaml --data $datapath --project $project --name $name --gaze cross --global-token --save-period $save_period
python eval_inference_vidhoi.py $project/$name/weights/last.pt --data $datapath --cfg configs/ablation_final/eval_hyp_f5.yaml --project $project --name $name --gaze cross --global-token
python eval_metrics_vidhoi.py $project/$name/eval/all_results.json --data $datapath

# f7 gaze ca + global token
name=f7_final
python train_vidhoi.py --epochs $epochs --warmup $warmup --cfg configs/final/train_hyp_f7.yaml --data $datapath --project $project --name $name --gaze cross --global-token --save-period $save_period
python eval_inference_vidhoi.py $project/$name/weights/last.pt --data $datapath --cfg configs/ablation_final/eval_hyp_f7.yaml --project $project --name $name --gaze cross --global-token
python eval_metrics_vidhoi.py $project/$name/eval/all_results.json --data $datapath
