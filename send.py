from datetime import datetime
import cv2
import uuid
import os
import argparse
import zerorpc
import requests
#%%
if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--img_path', type=str, default='', help='*.jpg path')
        opt = parser.parse_args()
        ts = datetime.utcnow()
        ff = cv2.imread(opt.img_path)
        msgtimestamp = ts.strftime("%Y-%m-%d %H:%M:%S.%f")

        c = zerorpc.Client()
        c.connect('tcp://api.insightglobe.net:4242')

        item = {'distance': 1,
                'vindex': 1,
                'hindex': 1,
                'timestamp': msgtimestamp,
                'imageextension': 'jpg',
                'lat': 22.372508,
                'lon': 114.003717,
                'vangle': '0000',
                'hangle': '0000',
                'sid': 'emulator1',
                'type': 'visevent',
                'autorun': True,
                'sensorid': 'emulator1',
                'dismiss': False,
                'eid': str(uuid.uuid4()),
                'camsitename': 'test1',
                'imgvis': cv2.imencode('.jpg', ff, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring(),
                'imgir': cv2.imencode('.jpg', ff, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tostring()
        }
        print(c.newEvent(item))
        print(opt.img_path + " sent")
