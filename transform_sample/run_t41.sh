cd ../
sed -ri 's/(target_device = ")[^"]*/\1T41/' models/common.py models/yolo.py
./detect.sh baby_nc2
./convert_onnx.sh baby_nc2 #baby_nc2 #coco_nc1
cd -
#CUDA_VISIBLE_DEVICES=0 python transform.py --model_file ../runs/train/yolov5s-person-4bit.onnx --output_file ./venus_sample_yolov5s/yolov5s_t41_magik.mk.h --config_file cfg/magik_t41.cfg
CUDA_VISIBLE_DEVICES=0 python transform.py --model_file ../weights/best.onnx --output_file ./out/yolov5s_t41_magik.mk.h --config_file cfg/magik_t41.cfg

#cd venus_sample_yolov5s
#cp makefile_files/Makefile_t41 Makefile
