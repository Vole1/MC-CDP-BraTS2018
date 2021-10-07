import os
import time
from abc import abstractmethod

import cv2
import numpy as np
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import Iterator, load_img, img_to_array


class BaseMaskDatasetIterator(Iterator):
    def __init__(self,
                 images_dir,
                 masks_dir,
                 image_ids,
                 images_paths,
                 input_channels,
                 crop_shape,
                 resize_shape,
                 preprocessing_function,
                 batch_size=8,
                 shuffle=True,
                 seed=None,
                 ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_ids = image_ids
        self.image_paths = images_paths
        self.input_channels = input_channels
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.preprocessing_function = preprocessing_function
        if seed is None:
            seed = np.uint32(time.time() * 1000)

        super(BaseMaskDatasetIterator, self).__init__(len(self.image_ids), batch_size, shuffle, seed)

    @abstractmethod
    def transform_mask(self, mask):
        raise NotImplementedError

    def get_output_shape(self):
        if self.crop_shape is not None and self.crop_shape != (None, None):
            return (*self.crop_shape, len(os.listdir(self.image_paths[0])))
        elif self.resize_shape is not None and self.resize_shape != (None, None):
            return (*self.resize_shape, len(os.listdir(self.image_paths[0])))
        else:
            path_to_img = os.path.join(self.image_paths[0], os.listdir(self.image_paths[0])[0])
            img_shape = (*np.array(img_to_array(load_img(path_to_img)), "uint8").shape, len(os.listdir(self.image_paths[0])))
            x0, y0 = 0, 0
            if (img_shape[1] % 32) != 0:
                x0 = (32 - img_shape[1] % 32)
            if (img_shape[0] % 32) != 0:
                y0 = (32 - img_shape[0] % 32)
            return (img_shape[0] + x0, img_shape[1] + y0, *img_shape[2:])

    def create_opencv_mask(self, mask_in):
        tmp = mask_in.copy()
        tmp = tmp.astype('uint8')

        threshold_level = 127  # Set as you need...
        _, binarized = cv2.threshold(tmp, threshold_level, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        msk1 = np.zeros_like(tmp, dtype='uint8')
        msk1 = cv2.drawContours(msk1, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
        msk = np.stack((mask_in, msk1))
        msk = np.moveaxis(msk, 0, -1)
        return msk

    def crop_mask_and_image(self, mask, image, crop_shape):
        if isinstance(crop_shape, int):
            crop_shape = (crop_shape, crop_shape)
        x0 = (image.shape[0] - crop_shape[0]) // 2
        x1 = (image.shape[0] - crop_shape[0]) - x0
        y0 = (image.shape[1] - crop_shape[1]) // 2
        y1 = (image.shape[1] - crop_shape[1]) - y0
        cropped_image = image[x0:x1, y0:y1, :]
        cropped_mask = mask[x0:x1, y0:y1, :]
        return cropped_mask, cropped_image

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []

        for batch_index, image_index in enumerate(index_array):

            img_path = self.image_paths[image_index]
            if img_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm')):
                image = np.array(img_to_array(load_img(img_path)), "uint8")
            elif img_path.endswith('.npy'):             # image-like array with more than 3 channels formatted - [W*H*C]
                image = np.load(img_path)
            else:
                raise ValueError("Unsupported type of image input data")

            mask_path = self.image_paths[image_index].replace(self.images_dir, self.masks_dir)
            if mask_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm')):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            elif mask_path.endswith('.npy'):
                mask = np.load(mask_path)
            else:
                raise ValueError("Unsupported type of mask input data")

            mask = self.create_opencv_mask(mask)
            if self.crop_shape is not None and self.crop_shape != (None, None):
                crop_mask, crop_image = self.crop_mask_and_image(mask, image, self.crop_shape)
                if len(np.shape(crop_mask)) == 2:
                    crop_mask = np.expand_dims(crop_mask, -1)
                #crop_mask = self.transform_mask(crop_mask)
                batch_x.append(crop_image)
                batch_y.append(crop_mask)
            elif self.resize_shape is not None and self.resize_shape != (None, None):
                resized_image = cv2.resize(image, tuple(self.resize_shape[j] for j in range(len(self.resize_shape))))
                resized_mask = cv2.resize(mask, tuple(self.resize_shape[j] for j in range(len(self.resize_shape))))
                batch_x.append(resized_image)
                #resized_mask = self.transform_mask(resized_mask)
                batch_y.append(resized_mask)
            else:
                x0, x1, y0, y1 = 0, 0, 0, 0
                if (image.shape[1] % 32) != 0:
                    x0 = int((32 - image.shape[1] % 32) / 2)
                    x1 = (32 - image.shape[1] % 32) - x0
                if (image.shape[0] % 32) != 0:
                    y0 = int((32 - image.shape[0] % 32) / 2)
                    y1 = (32 - image.shape[0] % 32) - y0
                image = np.pad(image, ((y0, y1), (x0, x1), (0, 0)), 'reflect')
                mask = np.pad(mask, ((y0, y1), (x0, x1), (0, 0)), 'reflect')
                batch_x.append(image)
                #mask = self.transform_mask(mask)
                batch_y.append(mask)
        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")
        t_x, t_y = self.preprocess_batch_x(batch_x), self.preprocess_batch_y(batch_y)
        return t_x, t_y

    def preprocess_batch_x(self, batch_x):
        if self.preprocessing_function and batch_x.shape[-1] == 3:
            return imagenet_utils.preprocess_input(batch_x, mode=self.preprocessing_function)

        if batch_x.shape[-1] > 3 and self.preprocessing_function == 'caffe':
            mean = [103.939, 116.779, 123.68]
            r_mean = mean[::-1]
            t_batch_x = batch_x[..., ::-1] - np.asarray((r_mean+r_mean)[:batch_x.shape[-1]])
            return t_batch_x
        else:
            print(f'Selected preprocessing function is not implemented for {batch_x.shape[-1]} channels!')
            print('Press enter to continue...', end='')
            input()
            return batch_x

    def preprocess_batch_y(self, batch_y, mode='WT'):
        if mode == 'WT':
            return np.where(batch_y > 1, 1, batch_y)
        else:
            raise NotImplementedError(f'Uknown mask preprocessing mode {mode}')

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)


