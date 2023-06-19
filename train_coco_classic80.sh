GPUS=1

#32bit:
python3 train.py \
    --data data/coco.yaml\
    --cfg models/yolov5s-classic-80.yaml \
    --weights '' \
    --batch-size 8 \
    --hyp data/hyps/hyp.scratch.yaml \
    --project ./runs/train/coco-32bit \
    --epochs 100 \
    --device 0

