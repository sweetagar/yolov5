export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
#GPUS=6
GPUS=1

#32bit:
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --data data/baby-classic4.yaml\
    --cfg models/yolov5s-classic-4.yaml \
    --weights weights/yolov5s-person-4bit.pt \
    --batch-size 2 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/baby-classic4-32bit \
    --epochs 300 \
    --device 0


#8bit:
#python3 -m torch.distributed.launch --nproc_per_node=$GPUS #--master_port=60051 train.py \
#     --data data/baby-head.yaml\
#     --cfg models/yolov5s.yaml \
#     --weights 'weights/best.pt' \
#     --batch-size 6 \
#     --hyp data/hyp.scratch-8bit.yaml \
#     --project ./runs/train/baby-head-8bit \
#     --epochs 300 \
#     --device 0

#4bit:
# python3 -m torch.distributed.launch --nproc_per_node=$GPUS #--master_port=60051 train.py \
#     --data data/baby-head.yaml\
#    --cfg models/yolov5s.yaml \
#     --weights 'weights/best.pt' \
#     --batch-size 6 \
#     --hyp data/hyp.scratch-4bit.yaml \
#     --project ./runs/train/baby-head-4bit \
#     --epochs 300 \
#     --device 0
#     --adam