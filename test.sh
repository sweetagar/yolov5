#python test.py --data data/coco-person.yaml --weights ./runs/train/yolov5s-person-4bit.pt --imgs 640 --device 0 --batch-size 6

python test.py --data data/baby-head.yaml --weights ./runs/train/baby-head-4bit/weights/best.pt --imgs 640 --device 0 --batch-size 6
