GT_ROOT=./data/sav_val/Annotations_6fps
PRED_ROOT=./data/sav_val/Annotations_6fps

python sav_evaluator.py --gt_root ${GT_ROOT}\
                       --pred_root ${PRED_ROOT}