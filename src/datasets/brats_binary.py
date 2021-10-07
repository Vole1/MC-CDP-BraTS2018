import os

import numpy as np
from datasets.base import BaseMaskDatasetIterator
from sklearn.model_selection import train_test_split


class DSB2018BinaryDataset:
    def __init__(self, images_dir, masks_dir, channels, seed=777):
        super().__init__()
        self.seed = seed
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.channels = channels
        np.random.seed(seed)
        self.train_ids, self.val_ids, self.train_paths, self.val_paths = self.generate_ids()
        print("Found {} train images".format(len(self.train_ids)))
        print("Found {} val images".format(len(self.val_ids)))

    def get_generator(self,
                      image_ids,
                      image_paths,
                      channels,
                      crop_shape,
                      resize_shape,
                      preprocessing_function='torch',
                      batch_size=16,
                      shuffle=True):
        return DSB2018BinaryDatasetIterator(
            self.images_dir,
            self.masks_dir,
            image_ids,
            image_paths,
            channels,
            crop_shape,
            resize_shape,
            preprocessing_function,
            batch_size,
            shuffle=shuffle,
            seed=self.seed)

    def get_generator_with_paths(self,
                      image_ids,
                      image_paths,
                      channels,
                      crop_shape,
                      resize_shape,
                      preprocessing_function='torch',
                      batch_size=16,
                      shuffle=True):
        return DSB2018BinaryDatasetIteratorWithPaths(
            self.images_dir,
            self.masks_dir,
            image_ids,
            image_paths,
            channels,
            crop_shape,
            resize_shape,
            preprocessing_function,
            batch_size,
            shuffle=shuffle,
            seed=self.seed)

    def train_generator(self,
                        crop_shape=(256, 256),
                        resize_shape=(256, 256),
                        preprocessing_function='torch',
                        batch_size=16):
        return self.get_generator(self.train_ids, self.train_paths, self.channels, crop_shape, resize_shape,
                                  preprocessing_function, batch_size, True)

    def val_generator(self, resize_shape=(256, 256), preprocessing_function='caffe', batch_size=1):
        return self.get_generator(self.val_ids, self.val_paths, self.channels, None, resize_shape,
                                  preprocessing_function, batch_size, False)

    def test_generator(self, resize_shape=(256, 256), preprocessing_function='caffe', batch_size=1):
        return self.get_generator(self.train_ids + self.val_ids, self.train_paths + self.val_paths, self.channels, None,
                                  resize_shape, preprocessing_function, batch_size, False)

    def test_ensemble_generator(self, resize_shape=(256, 256), preprocessing_function='caffe', batch_size=1):
        return self.get_generator_with_paths(self.train_ids + self.val_ids, self.train_paths + self.val_paths, self.channels, None,
                                  resize_shape, preprocessing_function, batch_size, False)


    def generate_ids(self):
        all_ids = next(os.walk(self.images_dir))[2]
        all_paths = list(map(lambda x: self.images_dir + '/' + x, all_ids))
        train_ids, val_ids, train_paths, val_paths = train_test_split(all_ids, all_paths, test_size=0.1,
                                                                      random_state=self.seed)
        return train_ids, val_ids, train_paths, val_paths


class DSB2018BinaryDatasetIterator(BaseMaskDatasetIterator):
    def __init__(self, images_dir,
                 masks_dir,
                 image_ids,
                 images_paths,
                 channels,
                 crop_shape,
                 resize_shape,
                 preprocessing_function,
                 batch_size=8,
                 shuffle=True,
                 seed=None):
        super().__init__(images_dir, masks_dir, image_ids, images_paths, channels, crop_shape, resize_shape,
                         preprocessing_function, batch_size, shuffle, seed)

    def transform_mask(self, mask):
        mask[mask > 127] = 255
        mask = np.clip(mask, 0, 255)
        return np.array(mask, "float32") / 255.


class DSB2018BinaryDatasetIteratorWithPaths(DSB2018BinaryDatasetIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        return super()._get_batches_of_transformed_samples(index_array),\
               [self.image_paths[image_index]+f'[{self.image_ids[image_index]}]' for batch_index, image_index in enumerate(index_array)]
