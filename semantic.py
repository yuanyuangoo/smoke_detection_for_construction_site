from scipy.io import loadmat
from PIL import Image
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from mit_semseg.config import cfg
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import csv
names = {}
with open('./segmentation/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def swap_key_value(my_dict):
    my_dict2 = {y: x for x, y in my_dict.items()}
    return my_dict2


def sem_mask(cfg_file='./segmentation/ade20k-resnet50dilated-ppm_deepsup.yaml', gpu=0, sources='video.txt'):
    cfg.merge_from_file(cfg_file)
    # cfg.merge_from_list(opts)
    # logger = setup_logger(distributed_rank=0)   # TODO
    # logger.info("Loaded configuration file {}".format(cfg_file))
    # logger.info("Running with config:\n{}".format(cfg))
    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()
    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(
            cfg.MODEL.weights_decoder), "checkpoint does not exitst!   "+cfg.MODEL.weights_decoder
    torch.cuda.set_device(gpu)
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)
    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.cuda()
    if sources.endswith('.txt') or sources.endswith('.ip'):
        with open(sources, 'r') as f:
            sources = [x.strip()
                       for x in f.read().splitlines() if len(x.strip())]
    cap = cv2.VideoCapture(sources[0])
    assert cap.isOpened(), 'Failed to open %s'
    get_first_frame, first_frame = cap.read()  # guarantee first frame
    assert get_first_frame and first_frame is not None, 'Failed to get first frame'
    cv2.imwrite('first_frame.jpg', first_frame)
    cfg.list_test = [{'fpath_img': x}
                     for x in os.listdir('.') if x == 'first_frame.jpg']
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=1,
        drop_last=True)
    torch.backends.cudnn.benchmark = True
    return seg(segmentation_module, loader_test, 0)


def seg(segmentation_module, loader, gpu):
    segmentation_module.eval()
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']
        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class,
                                 segSize[0], segSize[1])
            # scores = async_copy_to(scores, gpu)
            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)
                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp.cpu() / len(cfg.DATASET.imgSizes)
            _, pred = torch.max(scores, dim=1)
            names_ = swap_key_value(names)
            # pred = scores[0][names_['windowpane']-1] > 0.002
            pred = as_numpy(pred.squeeze(0).cpu())
            pred[pred == names_['sky']-1] = -1
            pred[pred == names_['tree']-1] = -1
            pred[pred == names_['person']-1] = -1
            pred[pred == names_['windowpane']-1] = -1
            pred = (pred != -1).astype(np.uint8)


        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )
        # exit()
    return pred


def visualize_result(data, pred, cfg):
    (img, info) = data
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)
    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))


colors = loadmat(
    './segmentation/color150.mat')['colors']
if __name__ == "__main__":
    models = sem_mask()

# %%
# import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.encoders import get_preprocessing_fn

# model = smp.Unet('deeplabv3', encoder_weights='imagenet',
#                  in_channels=3, decoder_attention_type='scse',)
# preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
