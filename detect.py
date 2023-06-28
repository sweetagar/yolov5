"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.yolo import Model

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.6,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):

    # ===================================== 1、初始化一些配置 =====================================
    # 是否保存预测后的图片 默认nosave=False 所以只要传入的文件地址不是以.txt结尾 就都是要保存预测后的图片的
    save_img = not nosave and not source.endswith('.txt')  # save inference images 的
    print(f'save_img={save_img}')
    
    #isnumeric是否摄像头路径，是否为txt结尾的，lower转为小写，startswith是否为rtsp开通的
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    print(f'webcam={webcam}')
    
    # Directories 新建保存结果文件夹 
    #increment_path递归添加文件夹 project='runs/detect' name=exp，如果相同名字就会变成exp,exp2,exp3    
    # 检查当前Path(project) / name是否存在 如果存在就新建新的save_dir 默认exist_ok=False 需要重建
    # 将原先传入的名字扩展成新的save_dir 如runs/detect/exp存在 就扩展成 runs/detect/exp1
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run

    # 如果需要save txt就新建save_dir / 'labels' 否则就新建save_dir
    # 默认save_txt=False 所以这里一般都是新建一个 save_dir(runs/detect/expn)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize 初始化日志信息
    set_logging()

    # 获取当前主机可用的设备
    device = select_device(device)
    
    # 如果设配是GPU 就使用half(float16)  包括模型半精度和输入图片半精度
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # ===================================== 2、载入模型和模型参数并调整模型 =====================================
    # 2.1、加载Float32模型
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    #加载模型权重 weights[0] = yolov5s.pt
    print(f'weights={weights[0]}')
    chkpt = torch.load(weights[0])
    
    #模型初始化
    aaaaa=chkpt['model'].yaml
    print(f'chkpt={aaaaa}')
    
    # 给模型分配stride属性
    model = Model(chkpt['model'].yaml, ch=3).cuda() #by ronsyao
    #model = Model(chkpt['model'].yaml, ch=3, nc=2).cuda()
    state_dict = chkpt['ema' if chkpt.get('ema') else 'model'].float().state_dict()  # to FP32
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 2.2、载入一些模型参数
    # stride: 模型最大的下采样率 [8, 16, 32] 所有stride一般为32
    #获取模型步长,可以理解为grid cell的尺寸，一般为stride=32
    stride = int(model.stride.max())  # model stride
    print("stride====",stride)

    # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    #获取模型类别名，person就是一类
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # 2.3、调整模型
    # 是否将模型从float32 -> float16  加速推理
    if half:
        model.half()  # to FP16

    # 是否加载二次分类模型
    # 这里考虑到目标检测完是否需要第二次分类，自己可以考虑自己的任务自己加上  但是这里默认是False的 我们默认不使用
    classify = False #不需要resnet50的权重
    if classify:
        modelc = load_classifier(name='resnet50', n=80)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # ===================================== 3、加载推理数据 =====================================
    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # ===================================== 4、推理前测试 =====================================
    # 这里先设置一个全零的Tensor进行一次前向推理 判断程序是否正常
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # ===================================== 5、正式推理 =====================================
    t0 = time.time()
    # path: 图片/视频的路径
    # img: 进行resize + pad之后的图片
    # img0s: 原尺寸的图片
    # vid_cap: 当读取图片时为None, 读取视频时为视频源
    for path, img, im0s, vid_cap in dataset:    
        # 5.1、处理每一张图片的格式
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0 # 归一化 0 - 255 to 0.0 - 1.0
        
        # 如果图片是3维(RGB) 就在前面添加一个维度1当中batch_size=1
        # 因为输入网络的图片需要是4为的 [batch_size, channel, w, h]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        #君正推理库用    
        #magik_input = img.cpu().numpy()
        #magik_input = np.transpose(magik_input, (0, 2, 3, 1)) * 255
        #magik_input = np.round(magik_input)
        #np.save("./transform_sample/out/magik_input.npy", magik_input)
        #magik_input_uint8 = magik_input.astype(np.uint8)
        #magik_input_shape = magik_input_uint8.shape
        #magik_input_path = "./transform_sample/out/magik_input_nhwc_%d_%d_%d_%d.bin" % (magik_input_shape[0], magik_input_shape[1], magik_input_shape[2], magik_input_shape[3])
        #magik_input_uint8.tofile(magik_input_path)

        # 5.2、对每张图片/视频进行前向推理
        t1 = time_synchronized()        
        # pred shape=[1, num_boxes, xywh+obj_conf+classes] = [1, 18900, 25]
        pred = model(img, augment=augment)[0]

        # 5.3、nms除去多余的框
        # Apply NMS  进行NMS
        # conf_thres: 置信度阈值
        # iou_thres: iou阈值
        # classes: 是否只保留特定的类别 默认为None
        # agnostic_nms: 进行nms是否也去除不同类别之间的框 默认False
        # max_det: 每张图片的最大目标个数 默认1000
        # pred: [num_obj, 6] = [5, 6]   这里的预测信息pred还是相对于 img_size(640) 的
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # 5.4、考虑进行二次分类
        # Apply Classifier  如果需要二次分类 就进行二次分类  一般是不需要的
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 5.5、后续保存或者打印预测信息
        # 对每张图片进行处理  将pred(相对img_size 640)映射回原图img0 size
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                # 但是大部分我们一般都是从LoadImages流读取本都文件中的照片或者视频 所以batch_size=1
                # p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                # s: 输出信息 初始为 ''
                # im0: 原始图片 letterbox + pad 之前的图片
                # frame: 初始为0  可能是当前图片属于视频中的第几帧？
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            # 当前图片路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
            p = Path(p)  # to Path
            # 图片/视频的保存路径save_path 如 runs\\detect\\exp8\\bus.jpg
            save_path = str(save_dir / p.name)  # img.jpg
            print("save_path:", save_path)
            # txt文件(保存预测框坐标)保存路径 如 runs\\detect\\exp8\\labels\\bus
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            
            # print string  输出信息  图片shape (w, h)
            s += '%gx%g ' % img.shape[2:]  # print string

            #  normalization gain gn = [w, h, w, h]  用于后面的归一化
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # imc: for save_crop 在save_crop中使用
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(5000)  # 1 millisecond 5s

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
