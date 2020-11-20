from additional import get_bounding_box, get_mask, split_with_mask, reverse_coords
import sys
sys.path.append("./yolov5")
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov5.utils.plots import plot_one_box
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer 
from yolov5.utils.datasets import LoadStreams, letterbox
from yolov5.models.experimental import attempt_load
from numpy import random
import torch.backends.cudnn as cudnn
import torch
import cv2
from pathlib import Path
import time
import argparse, os
from argparse import ArgumentParser
import shutil

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        "output", opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Initialize
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    scale = (1422, 800)
    refPt = get_bounding_box(source, scale)
    mask, split_points = get_mask(refPt, imgsz=512, dialate=True)

    mask=letterbox(mask,new_shape=(512))[0]

    view_img, save_img = True, False
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz)

    # Initialize model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # Eval mode
    model.to(device).eval()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    max_batch = 10
    frameID=0

    last_time = t0
    
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections

        # imgs = torch.from_numpy(imgs).to(device).half()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)     

        if mask is not None:
            img = img*torch.from_numpy(mask).half().to(device)
    
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        frameID += 1
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = Path(path[i]), '%g: ' % frameID, im0s[i].copy()

            save_path = str(Path(out) / Path(p).name).replace('.mp4', '.jpg').replace('.jpg', str(t)+'.jpg')
            txt_path = "output/"+str(frameID)

            # s += '%gx%g ' % imgs.shape[2:]  # print string

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                save_txt = True

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                sent_event = False

                s += '%g %ss, ' % (len(det), 'smoke')  # add to string
                this_time=time.time()
                if this_time-last_time > 30 or last_time == t0:
                    print(this_time-last_time)
                    last_time = this_time
                    sent_event = True
                    # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, time.time() - t))

                # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            save_path = "output/"+str(frameID)+".jpg"
            if len(pred) > 0:
                cv2.imwrite(save_path, im0)
            # if sent_event:
            #     Popen([r"C:\\Users\\yuany\\Anaconda3\\envs\\py2\\python.exe",  "utils\\send.py", r'--img_path', save_path], shell=True)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                # if is_wrong:
                cv2.imwrite(save_path, im0)
            # else:
            #     a=1
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../yolov5/runs/train/exp/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='video.txt', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
            # detect()
            strip_optimizer(opt.weights)
        else:
            detect()
