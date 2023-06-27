cd /nfs/HubLinux/AI/venus_sample_yolov5s/
./venus_yolov5s_bin_uclibc_release /nfs/HubLinux/AI/venus_sample_yolov5s/out/yolov5s_t41_magik.bin /nfs/HubLinux/AI/venus_sample_yolov5s/baby.jpg

cd /nfs/HubLinux/AI/venus_sample_yolov5s/
./venus_yolov5s_bin_uclibc_release ./out/yolov5s_t41_magik_person.bin ./bus.jpg

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/HubLinux/AI/venus_sample_yolov5s/lib/uclibc
cd /nfs/HubLinux/AI/venus_sample_yolov5s/
./venus_yolov5s_bin_uclibc_profile ./out/yolov5s_t41_magik.bin  ./baby.jpg


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/HubLinux/AI/venus_sample_yolov5s/lib/uclibc
cd /nfs/HubLinux/AI/venus_sample_yolov5s/debug_feature
./venus_yolov5s_bin_uclibc_debug /nfs/HubLinux/AI/venus_sample_yolov5s/out/yolov5s_t41_magik.bin /nfs/HubLinux/AI/venus_sample_yolov5s/out/magik_input_nhwc_1_640_640_3.bin