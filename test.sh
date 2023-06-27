#python test.py --data data/coco-person.yaml --weights ./runs/train/yolov5s-person-4bit.pt --imgs 640 --device 0 --batch-size 6



if [[ ${1} = "baby_nc2" ]]
then
  python test.py --data data/baby-nc2.yaml --weights ./runs/train/baby-nc2-4bit/weights/best.pt --imgs 640 --device 0 --batch-size 6
elif [[ ${1} = "coco_nc80" ]]
then
	#python test.py --data data/coco.yaml --weights ./runs/train/coco_nc80-32bit/weights/best.pt --imgs 640 --device 0 --batch-size 6
	python test.py --data data/coco.yaml --weights ./weights/yolov5s-person-4bit.pt --imgs 640 --device 0 --batch-size 6
else
	echo "./test.sh baby_nc2"
	echo "./test.sh coco_nc80"
fi