/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : inference.cc
 * Authors     : ffzhou
 * Create Time : 2022-07-16 09:22:44 (CST)
 * Description :
 *
 */
#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#include "./stb/drawing.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"
static const uint8_t color[3] = {0xff, 0, 0};

#include "venus.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include <memory.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>

#define TIME
#ifdef TIME
#include <sys/time.h>
#endif

#ifdef VENUS_PROFILE
#define RUN_CNT 10
#else
#define RUN_CNT 10
#endif

#define IS_ALIGN_64(x) (((size_t)x) & 0x3F)

using namespace std;
using namespace magik::venus;

struct PixelOffset {
    int top;
    int bottom;
    int left;
    int right;
};

uint8_t* read_bin(const char* path)
{
    std::ifstream infile;
	printf("path:%s\n", path);
    infile.open(path, std::ios::binary | std::ios::in);
    infile.seekg(0, std::ios::end);
    int length = infile.tellg();
	printf("length:%d\n", length);
    infile.seekg(0, std::ios::beg);
    uint8_t* buffer_pointer = new uint8_t[length];
    infile.read((char*)buffer_pointer, length);
    infile.close();
    return buffer_pointer;
}

std::vector<std::string> splitString(std::string srcStr, std::string delimStr,bool repeatedCharIgnored = false)
{
    std::vector<std::string> resultStringVector;
    std::replace_if(srcStr.begin(), srcStr.end(), [&](const char& c){if(delimStr.find(c)!=std::string::npos){return true;}else{return false;}}, delimStr.at(0));
    size_t pos=srcStr.find(delimStr.at(0));
    std::string addedString="";
    while (pos!=std::string::npos) {
        addedString=srcStr.substr(0,pos);
        if (!addedString.empty()||!repeatedCharIgnored) {
            resultStringVector.push_back(addedString);
        }
        srcStr.erase(srcStr.begin(), srcStr.begin()+pos+1);
        pos=srcStr.find(delimStr.at(0));
    }
    addedString=srcStr;
    if (!addedString.empty()||!repeatedCharIgnored) {
        resultStringVector.push_back(addedString);
    }
    return resultStringVector;
}

void check_pixel_offset(PixelOffset &pixel_offset){
    // 5 5 -> 6 4
    // padding size not is Odd number
    if(pixel_offset.top % 2 == 1){
        pixel_offset.top += 1;
        pixel_offset.bottom -=1;
    }
    if(pixel_offset.left % 2 == 1){
        pixel_offset.left += 1;
        pixel_offset.right -=1;
    }
}

void trans_coords(std::vector<magik::venus::ObjBbox_t> &in_boxes, PixelOffset &pixel_offset,float scale){
    
    printf("pad_x:%d pad_y:%d scale:%f \n",pixel_offset.left,pixel_offset.top,scale);
    for(int i = 0; i < (int)in_boxes.size(); i++) {
        in_boxes[i].box.x0 = (in_boxes[i].box.x0 - pixel_offset.left) / scale;
        in_boxes[i].box.x1 = (in_boxes[i].box.x1 - pixel_offset.left) / scale;
        in_boxes[i].box.y0 = (in_boxes[i].box.y0 - pixel_offset.top) / scale;
        in_boxes[i].box.y1 = (in_boxes[i].box.y1 - pixel_offset.top) / scale;
    }
}

void generateBBox(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h);
void generateBBox_manyclass(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h);

void vector_print(std::vector<int32_t> const &input)
{
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(i) << ' ';
    }
	printf("\n");
}

#include <stdio.h>

void save_pointer_data(void* data, size_t size, const char* filename) {
    // 打开文件以供写入
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("无法打开文件 %s\n", filename);
        return;
    }
    
    // 写入数据到文件
    size_t bytes_written = fwrite(data, 1, size, file);
    if (bytes_written != size) {
        printf("写入文件时发生错误\n");
    }
    
    // 关闭文件
    fclose(file);
}


// /nfs/HubLinux/AI/venus_sample_yolov5s/venus_yolov5s_bin_uclibc_release /nfs/HubLinux/AI/venus_sample_yolov5s/out/yolov5s_t41_magik.bin /nfs/HubLinux/AI/venus_sample_yolov5s/baby.jpg
//./venus_yolov5s_bin_uclibc_debug yolov5s-person-4bit.bin magik_input_nhwc_1_640_480_3.bin
//./venus_yolov5s_bin_uclibc_profile yolov5s-person-4bit.bin bus.jpg
//./venus_yolov5s_bin_uclibc_nmem yolov5s-person-4bit.bin bus.jpg
int main(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
            printf("warning: could not set CPU affinity, continuing...\n");
    }

#ifdef VENUS_DEBUG
    int ret = 0;
    if (argc != 3)
    {
        printf("%s model_path image_bin\n", argv[0]);
        exit(0);
    }
    std::string model_path = argv[1];
    std::string image_bin = argv[2];
    uint8_t* imagedata = read_bin(image_bin.c_str());
	std::vector<std::string> result_str = splitString(splitString(image_bin, ".")[0],"_");
    int vec_size = result_str.size();

    int n = atoi(result_str[vec_size - 4].c_str());
    int in_h = atoi(result_str[vec_size - 3].c_str());
    int in_w = atoi(result_str[vec_size - 2].c_str());
    int c = atoi(result_str[vec_size - 1].c_str());
    printf("image_bin shape:%d %d %d %d\n", n, in_h, in_w, c);

    std::unique_ptr<venus::Tensor> input;
    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }
    std::unique_ptr<venus::BaseNet> test_net;
    test_net = venus::net_create(TensorFormat::NHWC);
    ret = test_net->load_model(model_path.c_str());

    input = test_net->get_input(0);
    magik::venus::shape_t rgba_input_shape = input->shape();
    printf("model-->%d ,%d %d \n",rgba_input_shape[1], rgba_input_shape[2], rgba_input_shape[3]);
    input->reshape({1, in_h, in_w , 4});
    uint8_t *indata = input->mudata<uint8_t>();
    std::cout << "input shape:" << std::endl;
    printf("-->%d %d \n",in_h, in_w);
    int data_cnt = 1;
    for (auto i : input->shape()) 
    {
        std::cout << i << ",";
        data_cnt *= i;
    }
    std::cout << std::endl;

	for (int i = 0; i < in_h; i ++)
	{
		for (int j = 0; j < in_w; j++)
		{
			indata[i*in_w*4 + j*4 + 0] = imagedata[i*in_w*3 + j*3 + 0];
			indata[i*in_w*4 + j*4 + 1] = imagedata[i*in_w*3 + j*3 + 1];
			indata[i*in_w*4 + j*4 + 2] = imagedata[i*in_w*3 + j*3 + 2];
			indata[i*in_w*4 + j*4 + 3] = 0;
		}
	}

    test_net->run();

#else

    int ret = 0;
    if (argc != 3) {
        printf("%s model_path img_path\n", argv[0]);
        exit(0);
    }

    int ori_img_h = -1;
    int ori_img_w = -1;
    float scale = 1.0;
    int in_w = 640, in_h = 384;
    //int in_w = 416, in_h = 416;

    PixelOffset pixel_offset;
    std::unique_ptr<venus::Tensor> input;

    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }
    std::unique_ptr<venus::BaseNet> test_net;
    test_net = venus::net_create(TensorFormat::NHWC);

    std::string model_path = argv[1];
    ret = test_net->load_model(model_path.c_str());

    std::string image_path = argv[2];
    int comp = 0;

	//读取图片
    unsigned char *imagedata = stbi_load(argv[2], &ori_img_w, &ori_img_h, &comp, 3); // image format is bgra
	printf("ori_img_w:%d ori_img_h:%d\n", ori_img_w, ori_img_h);
	
    magik::venus::shape_t temp_inshape;
    temp_inshape.push_back(1);
    temp_inshape.push_back(ori_img_h);
    temp_inshape.push_back(ori_img_w);
    temp_inshape.push_back(4);
	printf("ori-img->inshape#####################\n");
	vector_print(temp_inshape);

#if 0
    save_pointer_data(imagedata, (ori_img_h*ori_img_w*3), "./image_data.rgb");
#endif	
	/*
	N - Batch = 一张图片Batch=1
	H - Height = 图像在竖直方向有多少像素
	W - Width = W 表示水平方向像素数
	C - Channel 黑白图像的通道数 C = 1，而 RGB 彩色图像的通道数 C = 3   bgra时c=4
	*/
	/* 注册一个名字为input_tensor的Tensor */
    venus::Tensor input_tensor(temp_inshape);//[1,h,w,4] 
    uint8_t *temp_indata = input_tensor.mudata<uint8_t>();

	//图片数据填充这个tensor [1,ori_img_h,ori_img_w,4] ,如果检测不到，可以看看是否rgb到bgr的问题
	for (int i = 0; i < ori_img_h; i ++) {
		for (int j = 0; j < ori_img_w; j++) {
			temp_indata[i*ori_img_w*4 + j*4 + 0] = imagedata[i*ori_img_w*3 + j*3 + 0];//r
			temp_indata[i*ori_img_w*4 + j*4 + 1] = imagedata[i*ori_img_w*3 + j*3 + 1];//g
			temp_indata[i*ori_img_w*4 + j*4 + 2] = imagedata[i*ori_img_w*3 + j*3 + 2];//b
			temp_indata[i*ori_img_w*4 + j*4 + 3] = 0;//a
		}
	}
	
    Img img = {
        .w = ori_img_w,
        .h = ori_img_h,
        .c = 3,
        .w_stride = ori_img_w*3,
        .data = imagedata
    };

//    magik::venus::memcopy((void*)temp_indata, (void*)(imagedata), src_size * sizeof(uint8_t));
	
    input = test_net->get_input(0);
    magik::venus::shape_t input_shape = input->shape();
    printf("model-->input_shape:[%d,%d,%d,%d] \n",input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    input->reshape({1, in_h, in_w , 4});
    float scale_x = (float)in_w/(float)ori_img_w;
    float scale_y = (float)in_h/(float)ori_img_h;
    scale = scale_x < scale_y ? scale_x:scale_y;  //min scale
    printf("scale---> %f\n",scale);
    int valid_dst_w = (int)(scale*ori_img_w);
    if (valid_dst_w % 2 == 1) {
        valid_dst_w = valid_dst_w + 1;
    }
    int valid_dst_h = (int)(scale*ori_img_h);
    if (valid_dst_h % 2 == 1) {
        valid_dst_h = valid_dst_h + 1;
    }
    int dw = in_w - valid_dst_w;
    int dh = in_h - valid_dst_h;    
    pixel_offset.top = int(round(float(dh)/2 - 0.1));
    pixel_offset.bottom = int(round(float(dh)/2 + 0.1));
    pixel_offset.left = int(round(float(dw)/2 - 0.1));
    pixel_offset.right = int(round(float(dw)/2 + 0.1));    
    check_pixel_offset(pixel_offset);
	
    printf("resize valid_dst, w:%d h %d\n",valid_dst_w,valid_dst_h);
    //printf("resize padding over: \n");
    printf("padding info top :%d bottom %d left:%d right:%d \n",pixel_offset.top,pixel_offset.bottom,pixel_offset.left,pixel_offset.right);


    magik::venus::BsExtendParam param;
    param.pad_val = 0;
    param.pad_type = magik::venus::BsPadType::SYMMETRY;
    param.in_layout = magik::venus::ChannelLayout::RGBA;
    param.out_layout = magik::venus::ChannelLayout::RGBA;
	/*
	* input: input tensor 		[1,ori_img_h,ori_img_w,4] 
	* output: output tensor		[1,in_h,in_w,4] 
	* param: resize param
	*/
	//input是根据test_net->get_input(0)赋值
    warp_resize(input_tensor, *input, &param);

#ifdef TIME
    struct timeval tv; 
    uint64_t time_last;
    double time_ms;
#endif

    for(int i = 0 ; i < RUN_CNT; i++){
#ifdef TIME
		gettimeofday(&tv, NULL);
		time_last = tv.tv_sec*1000000 + tv.tv_usec;
#endif
        test_net->run();

#ifdef TIME
		gettimeofday(&tv, NULL);
		time_last = tv.tv_sec*1000000 + tv.tv_usec - time_last;
		time_ms = time_last*1.0/1000;
		printf("test_net run time_ms:%fms\n", time_ms);
#endif
	}


#ifdef TIME
    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec;
#endif
	//把test_net转成out_res tensor,(19200+4800+1200)个预测框
    std::unique_ptr<const venus::Tensor> out0 = test_net->get_output(0);// 1/8层
    std::unique_ptr<const venus::Tensor> out1 = test_net->get_output(1);// 1/16层
    std::unique_ptr<const venus::Tensor> out2 = test_net->get_output(2);// 1/32层
    auto shape0 = out0->shape();
    auto shape1 = out1->shape();
    auto shape2 = out2->shape();
	printf("out0->shape#####################\n");
	vector_print(shape0);
	printf("out1->shape#####################\n");
	vector_print(shape1);
	printf("out2->shape#####################\n");
	vector_print(shape2);
    int shape_size0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];// [1, 80, 80, 18((5+num_class)x3)]	80 x 80 x3个预测框 19200
    int shape_size1 = shape1[0] * shape1[1] * shape1[2] * shape1[3];// [1, 40, 40, 18((5+num_class)x3)] 40 x 40 x3个预测框 4800
    int shape_size2 = shape2[0] * shape2[1] * shape2[2] * shape2[3];// [1, 20, 20, 18((5+num_class)x3)] 20 x 20 x3个预测框 1200
    venus::Tensor temp0(shape0);
    venus::Tensor temp1(shape1);
    venus::Tensor temp2(shape2);
    float* p0 = temp0.mudata<float>();
    float* p1 = temp1.mudata<float>();
    float* p2 = temp2.mudata<float>();
	//out0->data填充p0
    memcopy((void*)p0, (void*)out0->data<float>(), shape_size0 * sizeof(float));
    memcopy((void*)p1, (void*)out1->data<float>(), shape_size1 * sizeof(float));
    memcopy((void*)p2, (void*)out2->data<float>(), shape_size2 * sizeof(float));
   	//out_res 合并三个输出tensor
    std::vector<venus::Tensor> out_res;
    out_res.push_back(temp0);
    out_res.push_back(temp1);
    out_res.push_back(temp2);


    std::vector<magik::venus::ObjBbox_t>  output_boxes;
    output_boxes.clear();
	
	//bbox+nms
	generateBBox_manyclass(out_res, output_boxes, in_w, in_h);
	
	//scale为原图
    trans_coords(output_boxes, pixel_offset, scale);
	
#ifdef TIME
    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec - time_last;
    time_ms = time_last*1.0/1000;
	printf("post net time_ms:%fms\n", time_ms);
#endif


    for (int i = 0; i < int(output_boxes.size()); i++)  {
        auto person = output_boxes[i];
        printf("box:   ");
        printf("x0:%d ",(int)person.box.x0);
        printf("y0:%d ",(int)person.box.y0);
        printf("x1:%d ",(int)person.box.x1);
        printf("y1:%d ",(int)person.box.y1);
        printf("score:%.2f ",person.score);
		printf("class_id:[%d]",person.class_id);
        printf("\n");

        Point pt1 = {
            .x = (int)person.box.x0,
            .y = (int)person.box.y0
        };
        Point pt2 = {
            .x = (int)person.box.x1,
            .y = (int)person.box.y1
        };
        sample_draw_box_for_image(&img, pt1, pt2, color, 2);
    }

    stbi_write_bmp("result.bmp", ori_img_w, ori_img_h, 3, img.data);// w h
    free(img.data);

    ret = venus::venus_deinit();
    if (0 != ret) {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
    return 0;
#endif

}

void manyclass_nms(std::vector<magik::venus::ObjBbox_t> &input, std::vector<magik::venus::ObjBbox_t> &output, int classnums, int type, float nms_threshold) {
  int box_num = input.size();
  std::vector<int> merged(box_num, 0);
  std::vector<magik::venus::ObjBbox_t> classbuf;
  for (int clsid = 0; clsid < classnums; clsid++) {
    classbuf.clear();
    for (int i = 0; i < box_num; i++) {
      if (merged[i])
        continue;
      if(clsid!=input[i].class_id)
        continue;
      classbuf.push_back(input[i]);
      merged[i] = 1;

    }
    magik::venus::nms(classbuf, output, nms_threshold, magik::venus::NmsType::HARD_NMS);
  }
}


void generateBBox(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h)
{
  float person_threshold = 0.3;//0.3;
  int classes = 1;
  float nms_threshold = 0.6;
  std::vector<float> strides = {8.0, 16.0, 32.0};
  int box_num = 3;
  std::vector<float> anchor = {10,13,  16,30,  33,23, 30,61,  62,45,  59,119, 116,90,  156,198,  373,326};

  std::vector<magik::venus::ObjBbox_t>  temp_boxes;
  //偏移值转换为候选框
  venus::generate_box(out_res, strides, anchor, temp_boxes, img_w, img_h, classes, box_num, person_threshold, magik::venus::DetectorType::YOLOV5);
  //候选框再进行nms
  venus::nms(temp_boxes, candidate_boxes, nms_threshold); 
}

/*
* out_res 输入网络输出结果
* candidate_boxes：输出候选框 
*/
void generateBBox_manyclass(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h)
{
  float person_threshold = 0.3;//0.1;
  int classes = 2;//80;
  float nms_threshold = 0.6;
  std::vector<float> strides = {8.0, 16.0, 32.0};
  int box_num = 3;
  std::vector<float> anchor = {10,13,  16,30,  33,23, 30,61,  62,45,  59,119, 116,90,  156,198,  373,326};

  std::vector<magik::venus::ObjBbox_t>  temp_boxes;
  //输出temp_boxes
  venus::generate_box(out_res, strides, anchor, temp_boxes, img_w, img_h, classes, box_num, person_threshold, magik::venus::DetectorType::YOLOV5);
  printf("temp_boxes.size():%d\n", int(temp_boxes.size()));
  
  manyclass_nms(temp_boxes, candidate_boxes, classes, 0, nms_threshold);
  printf("candidate_boxes.size():%d\n", int(candidate_boxes.size()));

}

