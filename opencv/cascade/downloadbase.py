import urllib.request
import cv2
import numpy as np
import os
import subprocess


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


class CascadeImageProcessor(DownloadPath):

    def __init__(self, download_dir='downloads'):
        super().__init__(download_dir)

    def resize_image(self, image):
        resized = cv2.resize(image, (100, 100))
        return resized

    def grayscale_and_save(self, file_name):
        gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        resized = self.resize_image(gray)
        cv2.imwrite(file_name, resized)
        print('Resized Grayscale Image Saved')

    def remove_uglies(self):
        for sign_path in os.listdir(self.dirs['main']):
            if sign_path == 'uglies':
                continue
            for img in os.listdir(self.dirs[sign_path]):
                for ugly in os.listdir(self.dirs['uglies']):

                    try:
                        current_img_path = os.path.join(
                            self.dirs[sign_path], img)
                        ugly_img = cv2.imread(os.path.join(
                            self.dirs['uglies'], ugly))
                        current_img = cv2.imread(current_img_path)

                        if ugly_img.shape == current_img.shape and not (np.bitwise_xor(ugly_img, current_img).any()):
                            print('Ugly image found: {}'.format(
                                current_img_path))
                            print('Image removed')
                            os.remove(current_img_path)

                    except Exception as err:
                        print(str(err))

    def download_and_process(self, urls, pos=False, count=None):
        pic_count = count + 1
        base_url = self.dirs['neg'] if pos is False else self.dirs['pos']

        for image_url in urls.split('\n'):
            # if os.path.exists(base_url + str(pic_count) + '.jpg'):
            #     print(pic_count)
            #     pic_count += 1
            #     continue

            try:
                print('Downloading Image No {}: {}'.format(pic_count, image_url))
                urllib.request.urlretrieve(
                    image_url, base_url + str(pic_count) + '.jpg')
                self.grayscale_and_save(
                    base_url + str(pic_count) + '.jpg')
                pic_count += 1

            except Exception as err:
                print(str(err))

        return pic_count

    def prepare_store_images(self, neg_urls=['http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513', 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152'], pos_urls=''):
        last_neg = 0
        last_pos = 0

        for neg_url in neg_urls:
            neg_image_urls = urllib.request.urlopen(neg_url).read().decode()
            last_neg = self.download_and_process(
                neg_image_urls, count=last_neg)

        for pos_url in pos_urls:
            pos_image_urls = urllib.request.urlopen(pos_url).read().decode()
            last_pos = self.download_and_process(
                pos_image_urls, pos=True, count=last_pos)

    def create_desc_files(self):
        if os.path.exists('bg.txt'):
            os.remove('bg.txt')
        if os.path.exists('info.dat'):
            os.remove('info.dat')

        for sign_type in os.listdir(self.dirs['main']):
            if sign_type != 'neg' and sign_type != 'pos':
                continue
            else:
                for img in os.listdir(self.dirs[sign_type]):
                    if sign_type == 'neg':
                        line = os.path.join(self.dirs[sign_type], img) + '\n'
                        with open('bg.txt', 'a') as f:
                            f.write(line)

                    elif sign_type == 'pos':
                        line = os.path.join(
                            self.dirs[sign_type], img) + ' 1 0 0 50 50\n'
                        with open('info.dat', 'a') as f:
                            f.write(line)
