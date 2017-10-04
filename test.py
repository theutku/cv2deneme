import urllib.request
import cv2
import numpy as np
import os


class DownloadPath():

    def __init__(self, download_dir):
        self.download_dir = download_dir
        self.dirs = {
            'main': self.download_dir,
            'pos': os.path.join(self.download_dir, 'pos'),
            'neg': os.path.join(self.download_dir, 'neg'),
            'uglies': os.path.join(self.download_dir, 'uglies')
        }

        self._check_directories()

    def _check_directories(self):

        for key, value in self.dirs.items():
            if not os.path.exists(self.dirs[key]):
                os.makedirs(self.dirs[key])

    def create_pos_neg(self):
        for sign_type in os.listdir(self.dirs['main']):
            if sign_type != 'neg' and sign_type != 'pos':
                continue
            else:
                print(sign_type)
                for img in os.listdir(self.dirs[sign_type]):

                    line = os.path.join(self.dirs[sign_type], img) + '\n'
                    print(line)
                    if sign_type == 'neg':
                        with open(os.path.join(self.dirs['main'], 'bg.txt'), 'a') as f:
                            f.write(line)

                    elif sign_type == 'pos':
                        with open(os.path.join(self.dirs['main'], 'info.dat'), 'a') as f:
                            f.write(line)


cas = DownloadPath('test')
cas.create_pos_neg()
