from additional import get_bounding_box, get_mask, split_with_mask, reverse_coords
import sys
sys.path.append("./yolov5")
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov5.utils.plots import plot_one_box
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from yolov5.utils.datasets import LoadStreams, LoadImages
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
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
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
    
    for path, _, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections

        # imgs = torch.from_numpy(imgs).to(device).half()
        imgs=im0s[0].copy()/255.0
        imgs=imgs.transpose(2, 0, 1)
        imgs = torch.from_numpy(imgs).to(device).half()
        if imgs.ndimension() == 3:
            imgs = imgs.unsqueeze(0)        

        if mask is not None:
            imgs = imgs*torch.from_numpy(mask).to(device)
        
        imgs_split, split_points = split_with_mask(
            imgs, imgsz, refPt, split_points)
        preds=[]
        for i in range(int(len(imgs_split)/max_batch)+1):
            input_splits = imgs_split[i *
                                      max_batch:min((i+1)*max_batch, len(imgs_split))].half()
            # input_splits = imgs_split[4:5]
            # for idx, s in enumerate(imgs_split):
            #     tmp=np.float32(
            #         torch.Tensor.cpu(s).detach().numpy().transpose(1, 2, 0)*255)
            #     cv2.imwrite('tmp/'+str(idx)+'_f'+str(frameID)+'.jpg', tmp)

            if len(input_splits):
                preds.append(model(input_splits, augment=opt.augment)[0])
        preds = torch.cat(preds)
        if opt.half:
            preds = preds.float()

        # Apply NMS
        preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        frameID += 1
        # Apply Classifier
        if classify:
            preds = apply_classifier(preds, modelc, imgs, im0s)

        # Process detections
        n_all=0
        dets=[]
        for i, det in enumerate(preds):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, s, im0 = path[0], path[0]+'   ', imgs_split[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name).replace('.mp4', '.jpg').replace('.jpg', str(t)+'.jpg')
                
            # s += '%gx%g ' % imgs.shape[2:]  # print string

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                split_point= split_points[i]
                det[:, :4]=reverse_coords(det[:, :4], imgsz,split_point).round()
                # det[:, :4] = scale_coords(
                #     imgs.shape[2:], det[:, :4], im0.shape).round()
                det[:,1] = torch.clamp(det[:,1], refPt[0][1], refPt[1][1])
                det[:,3] = torch.clamp(det[:,3], refPt[0][1], refPt[1][1])
                det[:,0] = torch.clamp(det[:,0], refPt[0][0], refPt[1][0])
                det[:,2] = torch.clamp(det[:,2], refPt[0][0], refPt[1][0])
                det = det[det[:, 3]-det[:, 1] > 0][det[det[:, 3]-det[:, 1]
                                                       > 0][:, 2]-det[det[:, 3]-det[:, 1] > 0][:, 0] > 0]
                dets.append(det)

        sent_event = False
        if len(dets):
            dets = torch.cat(dets)

        # Print results
            for c in dets[:, -1].unique():
                n = (dets[:, -1] == c).sum()  # detections per class
                n_all += n

            # Write results
            for *xyxy, conf, cls in dets:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                if save_img or view_img:  # Add bbox to image
                    # label = '%s %.2f' % (names[int(cls)], conf)
                    label = '%s %.2f' % (r"19th floor", conf)
                    plot_one_box(xyxy, im0s[0], label=label, color=[255, 0, 0])


            if n_all:
                s += '%g %ss, ' % (n_all, 'smoke')  # add to string
                this_time=time.time()
                if this_time-last_time > 30 or last_time == t0:
                    print(this_time-last_time)
                    last_time = this_time
                    sent_event = True
                # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, time.time() - t))

                # Stream results
        if view_img:
            im_full = cv2.resize(im0s[0], scale)

            cv2.imshow(p, im_full)

            save_path = "output/"+str(frameID)+".jpg"

            if n_all > 0:
                cv2.imwrite(save_path, im_full)
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
    parser.add_argument('--output', type=str, default='./output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', default=True, action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', default=False,action='store_true', help='augmented inference')
    parser.add_argument('--update', default=False, action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
            # detect()
            strip_optimizer(opt.weights)
        else:
            detect()