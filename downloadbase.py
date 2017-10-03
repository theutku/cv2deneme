import urllib.request
import cv2
import numpy as np
import os


class CascadeImageDownloader():

    def __init__(self):
        self._check_directories()

    def _check_directories(self):
        if not os.path.exists('downloads'):
            os.makedirs('downloads')
        if not os.path.exists('downloads/neg'):
            os.makedirs('downloads/neg')
        if not os.path.exists('downloads/pos'):
            os.makedirs('downloads/pos')

    def _resize_image(self, image):
        resized = cv2.resize(image, (100, 100))
        return resized

    def _grayscale_and_save(self, file_name):
        gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        resized = self._resize_image(gray)
        cv2.imwrite(file_name, resized)
        print('Image Saved')

    def _download_and_process(self, urls, pos=False, count=None):
        pic_count = count + 1
        base_url = 'downloads/neg/' if pos is False else 'downloads/pos/'

        for image_url in urls.split('\n'):
            try:
                print(image_url)
                urllib.request.urlretrieve(
                    image_url, base_url + str(pic_count) + '.jpg')
                self._grayscale_and_save(
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
            last_neg = self._download_and_process(
                neg_image_urls, count=last_neg)

        for pos_url in pos_urls:
            pos_image_urls = urllib.request.urlopen(pos_url).read().decode()
            last_pos = self._download_and_process(
                pos_image_urls, pos=True, count=last_pos)

        # pic_count = 1
        # for image_url in neg_image_urls.split('\n'):
        #     try:
        #         print(image_url)
        #         urllib.request.urlretrieve(
        #             image_url, 'downloads/neg/' + str(pic_count) + '.jpg')
        #         self._grayscale_and_save(
        #             'downloads/neg/' + str(pic_count) + '.jpg')
        #         pic_count += 1

        #     except Exception as err:
        #         print(str(err))


cas = CascadeImageDownloader()
cas.prepare_store_images()
