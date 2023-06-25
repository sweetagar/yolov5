export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
GPUS=1

sed -i 's/bita = .*/bita = 4/g' models/common.py

if [[ ${1} = "baby_nc2" ]]
then
  sed -i 's/nc: .*/nc: 2/g' models/yolov5s.yaml
	python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --data data/baby-nc2.yaml\
    --cfg models/yolov5s.yaml \
    --weights './runs/train/baby-nc2-8bit/weights/best.pt' \
    --batch-size 8 \
    --hyp data/hyp.scratch-4bit.yaml \
    --project ./runs/train/baby-nc2-4bit \
    --epochs 300 \
    --device 0
elif [[ ${1} = "coco_nc80" ]]
then
  sed -i 's/nc: .*/nc: 80/g' models/yolov5s.yaml
	python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --data data/coco.yaml\
    --cfg models/yolov5s.yaml \
    --weights '' \
    --batch-size 8 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/coco-32bit \
    --epochs 300 \
    --device 0
else
	echo "./train_4bit.sh baby_nc2"
	echo "./train_4bit.sh coco_nc80"
fi



