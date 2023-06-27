#python detect.py --source data/images/bus.jpg --weights ./runs/train/yolov5s-person-4bit.pt  --imgs 640 --device 0 --view-img

#python detect.py --source data/images/bus.jpg --weights ./runs/train/yolov5s-person-4bit.pt  --imgs 640 --device 0 

#python detect.py --source data/images/zidane.jpg --weights ./runs/train/yolov5s-person-4bit/weights/best.pt --imgs 640 --device 0 

#python detect.py --source data/images/baby.jpg --weights ./runs/train/baby-head-32bit/weights/best.pt --img 640 --device 0 

#文件
#python detect.py --source data/images/baby.jpg --weights ./runs/train/baby-head-4bit/weights/best.pt --img 640 --device 0 --view-img

#视频
#python detect.py --source data/video/baby.mp4 --weights ./runs/train/baby-head-32bit/weights/best.pt --img 640 --device 0

#文件夹
#python detect.py --source ../datasets/baby_data/images/val/ --weights ./runs/train/baby-head-4bit/weights/best.pt --img 640 --device 0

#rtsp流
#python detect.py --weights ./runs/train/baby-head-4bit/weights/best.pt 'rtsp://192.168.0.105:554/0'

export MAGIK_TRAININGKIT_DUMP=1
export MAGIK_TRAININGKIT_PATH="./transform_sample/"

if [[ ${1} = "baby_nc2" ]]
then
	python detect.py --source data/images/baby.jpg --weights ./runs/train/baby-nc2-4bit/weights/best.pt --img 640 --device 0 --conf-thres 0.3 --iou-thres 0.6 --view-img
	#python detect.py --source data/video/baby.mp4 --weights ./runs/train/baby-nc2-4bit/weights/best.pt --img 640 --device 0
elif [[ ${1} = "baby_nc1" ]]
then
	python detect.py --source data/images/baby.jpg --weights ./runs/train/baby-nc1-4bit/weights/best.pt --img 640 --device 0 --view-img
elif [[ ${1} = "coco_nc80" ]]
then
	python detect.py --source data/images/bus.jpg --weights ./runs/train/coco_nc80-32bit/weights/best.pt --img 640 --device 0 --view-img
elif [[ ${1} = "coco_nc1" ]]
then
	python detect.py --source data/images/bus.jpg --weights ./weights/yolov5s-person-4bit-best.pt --img 640 --device 0 --view-img
else
  echo "./detect.sh baby_nc1"
	echo "./detect.sh baby_nc2"
	echo "./detect.sh coco_nc80"
	echo "./detect.sh coco_nc1"
fi
