cd ../
sed -ri 's/(target_device = ")[^"]*/\1T40/' models/common.py models/yolo.py
sh detect.sh
sh convert_onnx.sh
cd transform_sample
CUDA_VISIBLE_DEVICES=0 python transform.py --model_file ../runs/train/yolov5s-person-4bit.onnx --output_file ./venus_sample_yolov5s/yolov5s_t40_magik.mk.h --config_file cfg/magik_t40.cfg
cd venus_sample_yolov5s 
cp makefile_files/Makefile_t40 Makefile
