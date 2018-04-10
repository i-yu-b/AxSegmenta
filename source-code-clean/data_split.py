import glob
import random
import shutil

def data_split(imgs_path, masks_path, basic_path, train_ratio = 0.7, valid_ratio = 0.2, test_ratio = 0.1):
    images_filenames = glob.glob(imgs_path+'*.png')
    masks_filenames = glob.glob(masks_path+'*.png')
    paired_filenames = sorted(list(zip(images_filenames, masks_filenames)))
    # shuffle
    random.shuffle(paired_filenames)
    images_filenames, masks_filenames = zip(*paired_filenames)
    # split data
    total_examples = len(images_filenames)
    train_index = int(total_examples * train_ratio)
    valid_index = train_index + int(total_examples * valid_ratio)

    images_filenames_train, masks_filenames_train = images_filenames[:train_index],\
                                                    masks_filenames[:train_index]

    images_filenames_valid, masks_filenames_valid = images_filenames[train_index:valid_index],\
                                                    masks_filenames[train_index:valid_index]

    images_filenames_test, masks_filenames_test = images_filenames[valid_index:],\
                                                  masks_filenames[valid_index:]

    # save data
    for filename in images_filenames_train:
        shutil.copy2(filename, basic_path+'train/images/')
    for filename in masks_filenames_train:
        shutil.copy2(filename, basic_path+'train/masks/')

    for filename in images_filenames_valid:
        shutil.copy2(filename, basic_path+'valid/images/')
    for filename in masks_filenames_valid:
        shutil.copy2(filename, basic_path+'valid/masks/')

    for filename in images_filenames_test:
        shutil.copy2(filename, basic_path+'test/images/')
    for filename in masks_filenames_test:
        shutil.copy2(filename, basic_path+'test/masks/')

if __name__ == '__main__':
    data_split()
