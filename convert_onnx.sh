#python convert_onnx.py --weights ./runs/train/yolov5s-person-4bit.pt
#python convert_onnx.py --weights ./runs/train/yolov5s-person-4bit/weights/best.pt
#python convert_onnx.py --weights ./weights/best.pt


if [[ ${1} = "baby_nc2" ]]
then
  cp ./runs/train/baby-nc2-4bit/weights/best.pt ./weights/best.pt
	python convert_onnx.py --weights ./weights/best.pt
elif [[ ${1} = "baby_nc1" ]]
then
  cp ./runs/train/baby-nc1-4bit/weights/best.pt ./weights/best.pt
	python convert_onnx.py --weights ./weights/best.pt
elif [[ ${1} = "coco_nc80" ]]
then
  cp ./runs/train/coco_nc80-4bit/weights/best.pt ./weights/best.pt
  python convert_onnx.py --weights ./weights/best.pt
elif [[ ${1} = "coco_nc1" ]]
then
  cp ./runs/train/coco_nc1-4bit/weights/best.pt ./weights/best.pt
  python convert_onnx.py --weights ./weights/best.pt
else
	echo "./convert_onnx.sh baby_nc2"
	echo "./convert_onnx.sh coco_nc80"
	echo "./convert_onnx.sh coco_nc1"
fi