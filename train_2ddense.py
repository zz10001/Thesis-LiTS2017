import os
import numpy as np
import random
import argparse
from loss import weighted_cross_entropy_2d
from denseunet2d import DenseUNet
import preprocessing as pre
import keras.backend as k
from medpy.io import load
from multiprocessing.dummy import Pool as ThreadPool
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize

k.set_image_data_format('channels_last')
google_drive_path = pre.google_drive_path

parser = argparse.ArgumentParser(description='Keras 2D DenseUNet Training')

parser.add_argument('-data', type=str, default=google_drive_path + '/data', help='test images')
parser.add_argument('-save_path', type=str, default=google_drive_path + '/experiment')

parser.add_argument('-batch', type=int, default=40)
parser.add_argument('-input_size', type=int, default=224)
parser.add_argument('-model_weight', type=str, default='./model/densenet161_weights_tf.h5')
parser.add_argument('-input_column', type=int, default=3)

parser.add_argument('-mean', type=int, default=48)
parser.add_argument('-thread_num', type=int, default=14)
args = parser.parse_args()

mean = args.mean
thread_num = args.thread_num

liver_list = [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]


def data_augmentation(parameter_list):
    img = parameter_list[0]
    tumor = parameter_list[1]
    lines = parameter_list[2]
    num_id = parameter_list[3]
    min_index = parameter_list[4]
    max_index = parameter_list[5]

    #  Randomly scale --> cropping
    scale = np.random.uniform(0.8, 1.2)
    depth = int(args.input_size * scale)
    row = int(args.input_size * scale)
    column = 3

    sed = np.random.randint(1, num_id)
    cen = lines[sed - 1]
    cen = np.fromstring(cen, dtype=int, sep=' ')

    a = min(max(min_index[0] + depth / 2, cen[0]), max_index[0] - depth / 2 - 1)
    b = min(max(min_index[1] + row / 2, cen[1]), max_index[1] - row / 2 - 1)
    c = min(max(min_index[2] + column / 2, cen[2]), max_index[2] - column / 2 - 1)

    cropped_img = img[a - depth / 2:a + depth / 2, b - row / 2:b + row / 2, c - column / 2: c + column / 2 + 1].copy()
    cropped_tumor = tumor[a - depth / 2:a + depth / 2, b - row / 2:b + row / 2, c - column / 2:c + column / 2 + 1].copy()

    cropped_img -= mean

    # Randomly flipping --> mirroring
    flip_num = np.random.randint(0, 8)
    if flip_num == 1:
        cropped_img = np.flipud(cropped_img)
        cropped_tumor = np.flipud(cropped_tumor)
    elif flip_num == 2:
        cropped_img = np.fliplr(cropped_img)
        cropped_tumor = np.fliplr(cropped_tumor)
    elif flip_num == 3:
        cropped_img = np.rot90(cropped_img, k=1, axes=(1, 0))
        cropped_tumor = np.rot90(cropped_tumor, k=1, axes=(1, 0))
    elif flip_num == 4:
        cropped_img = np.rot90(cropped_img, k=3, axes=(1, 0))
        cropped_tumor = np.rot90(cropped_tumor, k=3, axes=(1, 0))
    elif flip_num == 5:
        cropped_img = np.fliplr(cropped_img)
        cropped_tumor = np.fliplr(cropped_tumor)
        cropped_img = np.rot90(cropped_img, k=1, axes=(1, 0))
        cropped_tumor = np.rot90(cropped_tumor, k=1, axes=(1, 0))
    elif flip_num == 6:
        cropped_img = np.fliplr(cropped_img)
        cropped_tumor = np.fliplr(cropped_tumor)
        cropped_img = np.rot90(cropped_img, k=3, axes=(1, 0))
        cropped_tumor = np.rot90(cropped_tumor, k=3, axes=(1, 0))
    elif flip_num == 7:
        cropped_img = np.flipud(cropped_img)
        cropped_tumor = np.flipud(cropped_tumor)
        cropped_img = np.fliplr(cropped_img)
        cropped_tumor = np.fliplr(cropped_tumor)

    cropped_tumor = resize(cropped_tumor, (args.input_size, args.input_size, args.input_cols), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    cropped_img = resize(cropped_img, (args.input_size, args.input_size, args.input_cols), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    return cropped_img, cropped_tumor[:, :, 1]


def generate_arrays_from_file(data_id, volume_list, segmentation_list, tumor_lines, liver_lines, tumor_id, liver_id, min_index_list, max_index_list):
    while 1:
        x = np.zeros((args.batch, args.input_size, args.input_size, args.input_column), dtype='float32')
        y = np.zeros((args.batch, args.input_size, args.input_size, 1), dtype='int16')
        parameter_list = []
        for _ in range(args.batch):
            count = random.choice(data_id)
            img = volume_list[count]
            tumor = segmentation_list[count]
            min_index = min_index_list[count]
            max_index = max_index_list[count]
            num = np.random.randint(0, 6)
            # To generate batches with 50% liver and 50% tumor CT scans
            if num < 3 or (count in liver_list):
                lines = liver_lines[count]
                num_id = liver_id[count]
            else:
                lines = tumor_lines[count]
                num_id = tumor_id[count]
            parameter_list.append([img, tumor, lines, num_id, min_index, max_index])
        pool = ThreadPool(thread_num)
        result_list = pool.map(data_augmentation, parameter_list)
        # result_list will look like cropped_img, cropped_tumor[:,:,1]
        pool.close()
        pool.join()

        for i in range(len(result_list)):
            x[i, :, :, :] = result_list[i][0]
            y[i, :, :, 0] = result_list[i][1]
        yield (x, y)


def load_fast_files():
    data_id = list(range(131))
    volume_list = []
    segmentation_list = []
    min_index_list = []
    max_index_list = []
    tumor_lines = []
    tumor_id = []
    liver_lines = []
    liver_id = []

    for i in range(131):
        volume, img_header = load(args.data + '/myTrainingData/volume-' + str(i) + '.nii')
        segmentation, tumor_header = load(args.data + '/myTrainingData/segmentation-' + str(i) + '.nii')
        volume_list.append(volume)
        segmentation_list.append(segmentation)

        max_min_list = np.loadtxt(args.data + '/myTrainingDataTxt/LiverBox/box_' + str(i) + '.txt', delimiter=' ')
        min_index = max_min_list[0:3]
        max_index = max_min_list[3:6]
        min_index = np.array(min_index, dtype='int')
        max_index = np.array(max_index, dtype='int')
        min_index[0] = max(min_index[0] - 3, 0)
        min_index[1] = max(min_index[1] - 3, 0)
        min_index[2] = max(min_index[2] - 3, 0)
        max_index[0] = min(volume.shape[0], max_index[0] + 3)
        max_index[1] = min(volume.shape[1], max_index[1] + 3)
        max_index[2] = min(volume.shape[2], max_index[2] + 3)
        min_index_list.append(min_index)
        max_index_list.append(max_index)
        f1 = open(args.data + '/myTrainingDataTxt/TumorPixels/tumor_' + str(i) + '.txt', 'r')
        tumor_line = f1.readlines()
        tumor_lines.append(tumor_line)
        tumor_id.append(len(tumor_line))
        f1.close()
        f2 = open(args.data + '/myTrainingDataTxt/LiverPixels/liver_' + str(i) + '.txt', 'r')
        liver_line = f2.readlines()
        liver_lines.append(liver_line)
        liver_id.append(len(liver_line))
        f2.close()

    return data_id, volume_list, segmentation_list, tumor_lines, liver_lines, tumor_id, liver_id, min_index_list, max_index_list


def train_and_predict():
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model = DenseUNet(reduction=0.5)
    model.load_weights(args.model_weight, by_name=True)
    # model = make_parallel(model, args.b / 10, mini_batch=10)
    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=[weighted_cross_entropy_2d])

    data_id, volume_list, segmentation_list, tumor_lines, liver_lines, tumor_id, liver_id, min_index_list, max_index_list = load_fast_files()

    print('-' * 30)
    print('Fitting model......')
    print('-' * 30)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists(args.save_path + "/model"):
        os.mkdir(args.save_path + '/model')
        os.mkdir(args.save_path + '/history')
    else:
        if os.path.exists(args.save_path + "/history/lossbatch.txt"):
            os.remove(args.save_path + '/history/lossbatch.txt')
        if os.path.exists(args.save_path + "/history/lossepoch.txt"):
            os.remove(args.save_path + '/history/lossepoch.txt')

    model_checkpoint = ModelCheckpoint(args.save_path + '/model/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=1)

    steps = 27386 / args.batch
    model.fit_generator(generate_arrays_from_file(data_id, volume_list, segmentation_list, tumor_lines, liver_lines, tumor_id, liver_id, min_index_list, max_index_list), steps_per_epoch=steps,
                        epochs=6000, verbose=1, callbacks=[model_checkpoint], max_queue_size=10,
                        workers=3, use_multiprocessing=True)

    print('Finished Training .......')


if __name__ == '__main__':
    train_and_predict()
