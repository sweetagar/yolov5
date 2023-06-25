export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
#GPUS=6
GPUS=1

cp ./models/common_4bit.py ./models/common.py

if [[ ${1} = "baby_nc2" ]]
then
#32bit:
#--resume ./runs/train/coco-32bit/weights/last.pt 
	python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --data data/baby-nc2.yaml\
    --cfg models/yolov5s-classic-80.yaml \
    --weights './runs/train/baby-nc2-8bit/weights/best.pt' \
    --batch-size 8 \
    --hyp data/hyp.scratch-4bit.yaml \
    --project ./runs/train/baby-nc2-4bit \
    --epochs 300 \
    --device 0
elif [[ ${1} = "coco_nc80" ]]
then
	python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --resume ./runs/train/coco-32bit/weights/last.pt \
    --data data/coco.yaml\
    --cfg models/yolov5s-classic-80.yaml \
    --weights '' \
    --batch-size 8 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/coco-32bit \
    --epochs 300 \
    --device 0
elif [[ ${1} = "coco_nc1" ]]
then
    python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --data data/coco-person.yaml\
    --cfg models/yolov5s-classic-1.yaml \
    --weights weights/best.pt \
    --batch-size 8 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/coco-person-32bit \
    --epochs 5 \
    --device 0
elif [[ ${1} = "coco128_nc80" ]]
then
	python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --data data/coco128.yaml\
    --cfg models/yolov5s-classic-80.yaml \
    --weights '/home/ubuntu/AI/yolov5-magic/runs/train/coco-32bit/weights/best.pt'\
    --batch-size 2 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/coco128-32bit \
    --epochs 300 \
    --device 0
elif [[ ${1} = "coco128_nc1" ]]
then
	python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --data data/coco128-persion.yaml\
    --cfg models/yolov5s.yaml \
    --weights weights/best.pt \
    --batch-size 32 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/coco128-persion-32bit \
    --epochs 300 \
    --device 0	    
else
	echo "./train.sh baby_nc2"
	echo "./train.sh coco_nc80"
	echo "./train.sh coco_nc1"
fi



