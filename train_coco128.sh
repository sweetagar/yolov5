export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
#GPUS=6
GPUS=1

#32bit:
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --data data/coco128.yaml\
    --cfg models/yolov5s-classic-80.yaml \
    --weights '/home/ubuntu/AI/yolov5-magic/runs/train/coco-32bit/weights/best.pt'\
    --batch-size 2 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/coco128-32bit \
    --epochs 300 \
    --device 0

#8bit:
#python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
#     --data data/coco-person.yaml\
#     --cfg models/yolov5s.yaml \
#     --weights './runs/train/yolov5s-person-32bit/weights/best.pt' \
#     --batch-size 8 \
#     --hyp data/hyp.scratch-8bit.yaml \
#     --project ./runs/train/yolov5s-person-8bit \
#     --epochs 300 \
#     --device 0


#4bit:
# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
#     --data data/coco-person.yaml\
#     --cfg models/yolov5s.yaml \
#     --weights './runs/train/yolov5s-person-8bit/weights/best.pt' \
#     --batch-size 8 \
#     --hyp data/hyp.scratch-4bit.yaml \
#     --project ./runs/train/yolov5s-person-4bit \
#     --epochs 300 \
#     --device 0
#     --adam