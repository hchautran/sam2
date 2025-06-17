
python training/scripts/sav_frame_extraction_submitit.py --sav-vid-dir \
   --sav-frame-sample-rate 24 \
   --output-dir ./data/sav_val/JPEGImages_24fps 


# val_path=/media/caduser/MyBook/chau/sam2/data/sav_val

# python ./tools/vos_inference.py \
#   --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
#   --sam2_checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
#   --base_video_dir $val_path/JPEGImages_24fps \
#   --input_mask_dir $val_path/Annotations_6fps \
#   --video_list_file $val_path/sav_val.txt \
#   --per_obj_png_file \
#   --output_mask_dir ./outputs/sav_val_pred_pngs