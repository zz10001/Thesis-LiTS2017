import os
import numpy as np
import argparse
from loss import weighted_cross_entropy
from hybridnet import dense_rnn_net
from denseunet3d import denseunet_3d
import preprocessing as pre
import keras.backend as k
from medpy.io import load
from multiprocessing.dummy import Pool as ThreadPool
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize

k.set_image_data_format('channels_last')
google_drive_path = pre.google_drive_path

parser = argparse.ArgumentParser(description='Keras H-DenseUNet Training')

parser.add_argument('-data', type=str, default=google_drive_path + '/data', help='test images')
parser.add_argument('-save_path', type=str, default=google_drive_path + '/experiments')

parser.add_argument('-batch', type=int, default=1)
parser.add_argument('-input_size', type=int, default=224)
parser.add_argument('-model_weight', type=str, default=google_drive_path+'/model/model_best.hdf5')
parser.add_argument('-input_cols', type=int, default=8)
parser.add_argument('-arch', type=str, default='')
parser.add_argument('-mean', type=int, default=48)

args = parser.parse_args()

thread_num = 14
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
    deps = int(args.input_size * scale)
    rows = int(args.input_size * scale)
    cols = args.input_cols

    sed = np.random.randint(1, num_id)
    cen = lines[sed - 1]
    cen = np.fromstring(cen, dtype=int, sep=' ')

    a = min(max(min_index[0] + deps / 2, cen[0]), max_index[0] - deps / 2 - 1)
    b = min(max(min_index[1] + rows / 2, cen[1]), max_index[1] - rows / 2 - 1)
    c = min(max(min_index[2] + cols / 2, cen[2]), max_index[2] - cols / 2 - 1)

    cropped_img = img[a - deps / 2:a + deps / 2, b - rows / 2:b + rows / 2, c - args.input_cols / 2: c + args.input_cols / 2].copy()
    cropped_tumor = tumor[a - deps / 2:a + deps / 2, b - rows / 2:b + rows / 2, c - args.input_cols / 2:c + args.input_cols / 2].copy()

    cropped_img -= args.mean

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
    #
    cropped_tumor = resize(cropped_tumor, (args.input_size, args.input_size, args.input_cols), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    cropped_img = resize(cropped_img, (args.input_size, args.input_size, args.input_cols), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    return cropped_img, cropped_tumor


def generate_arrays_from_file(batch_size, data_id, img_list, tumor_list, tumor_lines, liver_lines, tumor_id, liver_id, min_index_list, max_index_list):
    while 1:
        x = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols, 1), dtype='float32')
        y = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols, 1), dtype='int16')
        parameter_list = []
        for _ in range(batch_size):
            count = np.random.choice(data_id)
            img = img_list[count]
            tumor = tumor_list[count]
            min_index = min_index_list[count]
            max_index = max_index_list[count]

            num = np.random.randint(0, 6)
            if num < 3 or (count in liver_list):
                lines = liver_lines[count]
                num_id = liver_id[count]
            else:
                lines = tumor_lines[count]
                num_id = tumor_id[count]
            parameter_list.append([img, tumor, lines, num_id, min_index, max_index])
        pool = ThreadPool(thread_num)
        result_list = pool.map(data_augmentation, parameter_list)
        pool.close()
        pool.join()

        for i in range(len(result_list)):
            x[i, :, :, :, 0] = result_list[i][0]
            y[i, :, :, :, 0] = result_list[i][1]

        if np.sum(y == 0) == 0:
            continue
        if np.sum(y == 1) == 0:
            continue
        if np.sum(y == 2) == 0:
            continue

        yield (x, y)


def train_and_predict():
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    if args.arch == "3dpart":
        model = denseunet_3d(args)
        model_path = "/3dpart_model"
        sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=[weighted_cross_entropy])
        model.load_weights(args.model_weight, by_name=True, by_gpu=True, two_model=True, by_flag=True)
    else:
        model = dense_rnn_net(args)
        model_path = "/hybrid_model"
        sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=[weighted_cross_entropy])
        model.load_weights(args.model_weight)

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

        max_min = np.loadtxt(args.data + '/myTrainingDataTxt/LiverBox/box_' + str(i) + '.txt', delimiter=' ')
        min_index = max_min[0:3]
        max_index = max_min[3:6]
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

    if not os.path.exists(args.save_path + model_path):
        os.mkdir(args.save_path + model_path)
    if not os.path.exists(args.save_path + "/history"):
        os.mkdir(args.save_path + '/history')
    else:
        if os.path.exists(args.save_path + "/history/lossbatch.txt"):
            os.remove(args.save_path + '/history/lossbatch.txt')
        if os.path.exists(args.save_path + "/history/lossepoch.txt"):
            os.remove(args.save_path + '/history/lossepoch.txt')

    model_checkpoint = ModelCheckpoint(args.save_path + model_path + '/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=1,
                                       save_best_only=False, save_weights_only=False, mode='min', period=1)
    print('-' * 30)
    print('Fitting model......')
    print('-' * 30)

    steps = 27386 / (args.batch * 6)
    model.fit_generator(generate_arrays_from_file(args.batch, data_id, volume_list, segmentation_list, tumor_lines, liver_lines,
                        tumor_id, liver_id, min_index_list, max_index_list), steps_per_epoch=steps, epochs=6000, verbose=1, callbacks=[model_checkpoint], max_queue_size=10,
                        workers=3, use_multiprocessing=True)

    print('Finished Training .......')


if __name__ == '__main__':
    train_and_predict()
