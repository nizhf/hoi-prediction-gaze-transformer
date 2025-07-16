# Dataset Preparation
The dataset folder will look like the follows:  
```
    .  
    ├── VidOR  
    │   ├── video                       # can be deleted  
    │   ├── labels  
    │   ├── images  
    │   ├── VidHOI_annotation  
    │   ├── VidHOI_detection  
    │   ├── VidHOI_gaze  
    │   ├── yolov5_train_manually.txt   # only if you train the YOLOv5 detector
    │   └── yolov5_val_manually.txt     # only if you train the YOLOv5 detector
    └── action_genome  
        ├── videos                      # can be deleted  
        ├── frames  
        ├── annotations  
        └── action_genome_gaze  
```
## VidHOI
Following the original repository: https://github.com/coldmanck/VidHOI  
1. Download Training set and Validation set from VidOR https://xdshang.github.io/docs/vidor.html, then unzip the videos to `somewhere/VidOR/video/`.
2. Download the VidHOI annotation files `train_frame_annots.json` from https://drive.google.com/drive/folders/0B94KbEj1tO9nfi05YmlvS1ctdWRxVVhISVQ5b2FQN293MTRhMzFNY1p4QWY0TzhQVkR4ekk?resourcekey=0-TwBoRuVnDeRZCpgzACYPsA. If you want to use their object detection result, also download `det_val_frame_annots.json`. And download `val_frame_annots.json`, `obj_categories.json`, `pred_categories.json` from https://github.com/coldmanck/VidHOI/blob/master/slowfast/datasets/vidor-github/. Put all of them in `somewhere/VidOR/VidHOI_annotation/`.  
3. Copy `vidhoi_related/pred_split_categories.json` from this repo to `somewhere/VidOR/VidHOI_annotation/`.  
4. Copy `vidhoi_related/extract_vidor_frames.sh` from this repo to `somewhere/VidOR/`. Use this script to extract frames to `somewhere/images/`. You need ffmpeg to execute this script.  
## Action Genome
1. Download videos, annotations and toolkit from https://www.actiongenome.org/#download. Unzip videos to `somewhere/action_genome/videos/`, put annotations into `somewhere/action_genome/annotations/`. 
2. Use the `dump_frames.py` in the toolkit to extract frames to `somewhere/action_genome/frames/`. You should pass the correct path to the script. You need ffmpeg to execute this script.  
 
# Object Detection and Gaze Features
1. Download our YOLOv5 weights `vidor_yolov5l.pt` from [this link](https://tumde-my.sharepoint.com/:f:/g/personal/zhifan_ni_tum_de/Ev6sVnE0y2VBnmJ4RD65W7EB9jv0GlxkjgKmalvWMYwEDA), put it in `weights/yolov5/`
2. Use `misc_scripts/vidhoi_yolov5_deepsort_gaze_extraction.ipynb` and `misc_scripts/vidhoi_gt_gaze_extraction.ipynb` to extract gaze features for training and validation on VidHOI dataset. You may need to modify the dataset path and output path
3. The script will automatically download some weights for DeepSORT and Gaze Following.
4. Similarly, use `misc_scripts/ag_gt_gaze_extraction.ipynb` to extract gaze features for Action Genome dataset.  

### Optional: train the YOLOv5 detector
1. Use `misc_scripts/vidhoi_anno_to_yolov5.ipynb` to extract YOLOv5 annotations. We train the YOLOv5 model only on manually labeled frames.  
2. Run `python modules/object_tracking/yolov5/train.py --img 640 --batch 8 --epochs 50 --data configs/vidor.yaml --weights weights/yolov5/yolov5l.pt --save-period 1 --hyp configs/train_yolov5_finetune.yaml`  
3. You can interrupt the training when you feel the model is overfitted.  

# Training
An example of the workspace folder, with `$project` set to `../runs/sttran_gaze_vidhoi` for VidHOI and `../runs/sttran_gaze_ag` for Action Genome, `$name` set to `vidhoi_f0_final` and `ag_f0_final`.
```
    .
    ├── hoi-prediction-gaze-transformer  # This repo
    └── runs
        ├── sttran_gaze_vidhoi
        │   └── vidhoi_f0_final
        │       ├── eval
        │       │   ├── all_results.json
        │       │   ├── all_metrics.csv
        │       │   ├── human_centric_metrics.csv
        │       │   └── more log files
        │       ├── eval_det_yolov5l_deepsort
        │       │   └── similar to eval
        │       ├── wandb
        │       ├── weights
        │       │   └── last.pt
        │       └── more log files
        └── sttran_gaze_ag
            └── ag_f0_final
                └── similar to vidhoi_f0_final
``` 
## HOI detection/anticipation on VidHOI
Train with the best hyperparameter:  
`python train_vidhoi.py --epochs 25 --warmup 3 --cfg configs/final/train_hyp_fx.yaml --data $datapath --gaze cross --global-token --project $project --name vidhoi_fx_final --save-period 1`  
You should replace "fx" by "f0" for detection, and "f1", "f3", "f5", "f7" for anticipation. Also, replace those `$param` to your dataset path `somewhere/VidOR/` and desired output path. You may check `misc_script/final_exp_vidhoi.sh` as an example.  
## HOI detection on Action Genome
Train with the best hyperparameter:  
`python train_ag.py --epochs 25 --warmup 3 --cfg configs/final/train_hyp_f0.yaml --data $datapath --gaze cross --global-token --project $project --name ag_f0_final --save-period 1`  
You should replace those `$param` to your dataset path and desired output path. You may check `misc_script/final_exp_ag.sh` as an example.  

# Evaluation
You can use the model weights trained by yourself, or download our best model from [the same link as before](https://tumde-my.sharepoint.com/:f:/g/personal/zhifan_ni_tum_de/Ev6sVnE0y2VBnmJ4RD65W7EB9jv0GlxkjgKmalvWMYwEDA), put them to `$project/$name/weights/`. Replace $project and $run_name to you desired output path. Different future time should have different $name (see [Training](#training) section).  
Or we also provide all our result JSON files [here](https://tumde-my.sharepoint.com/:f:/g/personal/zhifan_ni_tum_de/Es96DJ9SPRlOqfqqkdfQQrEBy4Z-4nSN40xnu3rh6_N8Ng). You can then skip to [Compute Metrics](#compute-metrics) section.
## On VidHOI
### Generate Inference Results
Oracle mode:  
`python eval_inference_vidhoi.py $project/$name/weights/last.pt --data $datapath --cfg configs/final/eval_hyp_fx.yaml --project $project --name $name --gaze cross --global-token`  
Detection mode:  
`python eval_inference_vidhoi.py $project/$name/weights/last.pt --data $datapath --cfg configs/final/eval_hyp_fx.yaml --project $project --name $name --gaze cross --global-token --detection yolov5l_deepsort`  
If the weight is downloaded, change the `last.pt` to the corresponding file name. You should replace "fx" by "f0" for detection, and "f1", "f3", "f5", "f7" for anticipation. Also, replace those `$param` to your dataset path and desired output path. You may check `misc_script/final_exp_vidhoi.sh` as an example.  
### Compute Metrics
Oracle mode:  
`python eval_metrics_vidhoi.py $project/$name/eval/all_results.json --data $datapath`
Detection mode:  
`python eval_metrics_vidhoi.py $project/$name/eval_det_yolov5l_deepsort/all_results.json --data $datapath`
## PredCLS mode On Action Genome
### Generate Inference Results
`python eval_inference_ag.py $project/$name/weights/last.pt --data $datapath --cfg configs/final/eval_hyp_f0.yaml --project $project --name $name --gaze cross --global-token`
### Compute Metrics
`python eval_metrics_ag.py $project/$name/eval/all_results.json --data $datapath`
