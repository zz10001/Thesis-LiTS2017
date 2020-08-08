import os
import numpy as np
import tensorflow as tf
import time
import random
import keras.backend as k
import preprocessing as pre
from medpy.io import load
from multiprocessing.dummy import Pool as ThreadPool
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, ZeroPadding2D, concatenate, add
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from skimage.transform import resize
from custom_layers import Scale

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
k.set_image_data_format('channels_last')
path = pre.google_drive_path + '/result_train_denseU167_fast_new/'
data_path = pre.google_drive_path + '/data'

# Training variables
batch_size = 10
img_depth = 512
img_row = 512
img_column = 3
std = 37
thread_num = 14
text_file = 'myTrainingDataTxt'
mean = 48
concat_axis = 0

# List of CT scans that have only liver but not tumor
liver_list = [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]


# Data augmentation with cropping and flipping
def data_augmentation(parameter_list):
    img = parameter_list[0]
    tumor = parameter_list[1]
    lines = parameter_list[2]
    num_id = parameter_list[3]
    min_index = parameter_list[4]
    max_index = parameter_list[5]
    #  Randomly scale --> cropping
    scale = np.random.uniform(0.8, 1.2)
    depth = int(img_depth * scale)
    row = int(img_row * scale)
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
    flip_num = np.random.randint(0, 3)
    if flip_num == 1:
        cropped_img = np.flipud(cropped_img)
        cropped_tumor = np.flipud(cropped_tumor)
    elif flip_num == 2:
        cropped_img = np.fliplr(cropped_img)
        cropped_tumor = np.fliplr(cropped_tumor)

    cropped_tumor = resize(cropped_tumor, (img_depth, img_row, img_column), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    cropped_img = resize(cropped_img, (img_depth, img_row, img_column), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    return cropped_img, cropped_tumor[:, :, 1]


def generate_arrays_from_file(data_id, volume_list, segmentation_list, tumor_lines, liver_lines, tumor_id, liver_id, min_index_list, max_index_list):
    while 1:
        x = np.zeros((batch_size, img_depth, img_row, img_column), dtype='float32')
        y = np.zeros((batch_size, img_depth, img_row, 1), dtype='int16')
        parameter_list = []
        for _ in range(batch_size):
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


def weighted_cross_entropy(y_true, y_predict):
    y_predict_f = k.reshape(y_predict, (batch_size * img_depth * img_row, 3))
    y_true_f = k.reshape(y_true, (batch_size * img_depth * img_row,))

    soft_predict_f = k.softmax(y_predict_f)

    soft_predict_f = k.log(tf.clip_by_value(soft_predict_f, 1e-10, 1.0))

    neg = k.equal(y_true_f, k.zeros_like(y_true_f))

    neg_loss = tf.gather(soft_predict_f[:, 0], tf.where(neg))

    pos1 = k.equal(y_true_f, k.ones_like(y_true_f))
    pos1_loss = tf.gather(soft_predict_f[:, 1], tf.where(pos1))

    pos2 = k.equal(y_true_f, 2 * k.ones_like(y_true_f))
    pos2_loss = tf.gather(soft_predict_f[:, 2], tf.where(pos2))

    # 0.78 = weight for background, 0.65 = liver, and 8.57 = tumor
    loss = -k.mean(tf.concat([0.78 * neg_loss, 0.65 * pos1_loss, 8.57 * pos2_loss], 0))

    return loss


def DenseUNet(nb_dense_block=4, growth_rate=48, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, weights_path=None):
    """Instantiate the DenseNet 161 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    """
    eps = 1.1e-5

    compression = 1.0 - reduction

    global concat_axis

    # Handle Dimension Ordering (input dimension) for different backends
    # If Keras is on Tensorflow backends
    if k.image_data_format() == 'channels_last':
        concat_axis = 3
        img_input = Input(batch_shape=(batch_size, img_depth, img_row, 3), name='data')
    else:
        # Keras is on Theano backends
        concat_axis = 1
        # Then the channel dimension is in the first position instead
        img_input = Input(shape=(3, 224, 224), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [6, 12, 36, 24]  # For DenseNet-161
    box = []

    x = ZeroPadding2D((3, 3), name='convolution_1_zero_padding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='convolution_1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='convolution_1_batch_normalization')(x)
    x = Scale(axis=concat_axis, name='convolution_1_scale')(x)
    x = Activation('relu', name='convolution_1_ReLU')(x)

    box.append(x)

    x = ZeroPadding2D((1, 1), name='pooling_1_zero_padding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pooling_1')(x)

    for block_id in range(nb_dense_block - 1):
        stage = block_id + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_id], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        box.append(x)

        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        # Compression is added to further reduce the number of feature maps
        nb_filter = int(nb_filter * compression)

    final_stage = nb_dense_block
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='convolution' + str(final_stage) + '_blk_batch_normalization')(x)
    x = Scale(axis=concat_axis, name='convolution' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='ReLU' + str(final_stage) + '_blk')(x)

    box.append(x)

    up0 = UpSampling2D(size=(2, 2))(x)
    line0 = Conv2D(2208, (1, 1), padding="same", kernel_initializer="normal", name="line_0")(box[3])
    up0_sum = add([line0, up0])
    conv_up0 = Conv2D(768, (3, 3), padding="same", kernel_initializer="normal", name="upsampling_0")(up0_sum)
    bn_up0 = BatchNormalization(name="batch_normalization_upsampling_0")(conv_up0)
    ac_up0 = Activation('relu', name='ReLU_upsampling_0')(bn_up0)

    up1 = UpSampling2D(size=(2, 2))(ac_up0)
    up1_sum = add([box[2], up1])
    conv_up1 = Conv2D(384, (3, 3), padding="same", kernel_initializer="normal", name="upsampling_1")(up1_sum)
    bn_up1 = BatchNormalization(name="batch_normalization_upsampling_1")(conv_up1)
    ac_up1 = Activation('relu', name='ReLU_upsampling_1')(bn_up1)

    up2 = UpSampling2D(size=(2, 2))(ac_up1)
    up2_sum = add([box[1], up2])
    conv_up2 = Conv2D(96, (3, 3), padding="same", kernel_initializer="normal", name="upsampling_2")(up2_sum)
    bn_up2 = BatchNormalization(name="batch_normalization_upsampling_2")(conv_up2)
    ac_up2 = Activation('relu', name='ReLU_upsampling_2')(bn_up2)

    up3 = UpSampling2D(size=(2, 2))(ac_up2)
    up3_sum = add([box[0], up3])
    conv_up3 = Conv2D(96, (3, 3), padding="same", kernel_initializer="normal", name="upsampling_3")(up3_sum)
    bn_up3 = BatchNormalization(name="batch_normalization_upsampling_3")(conv_up3)
    ac_up3 = Activation('relu', name='ReLU_upsampling_3')(bn_up3)

    up4 = UpSampling2D(size=(2, 2))(ac_up3)
    conv_up4 = Conv2D(64, (3, 3), padding="same", kernel_initializer="normal", name="upsamping_4")(up4)
    conv_up4 = Dropout(rate=0.3)(conv_up4)
    bn_up4 = BatchNormalization(name="batch_normalization_upsampling_4")(conv_up4)
    ac_up4 = Activation('relu', name='ReLU_upsampling_4')(bn_up4)

    # Convolution 2
    x = Conv2D(3, (1, 1), padding="same", kernel_initializer="normal", name="dense167_classifier")(ac_up4)

    model = Model(img_input, x, name='denseunet_161')

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True, weight_decay=1e-4):
    """ Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    """

    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_' + str(stage) + '_' + str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def conv_block(x, stage, branch, nb_filter, dropout_rate=None):
    """Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    """
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_batch_normalization')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base + '_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_batch_normalization')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zero_padding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base + '_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1e-4):
    """ Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    """

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_batch_normalization')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


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

    time_stamp_1 = time.time()

    for idx in range(131):
        volume, img_header = load(data_path + '/myTrainingData/volume-' + str(idx) + '.nii')
        segmentation, tumor_header = load(data_path + '/myTrainingData/segmentation-' + str(idx) + '.nii')
        volume_list.append(volume)
        segmentation_list.append(segmentation)

        max_min = np.loadtxt(data_path + str(text_file) + '/LiverBox/box_' + str(idx) + '.txt', delimiter=' ')
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

        f1 = open(data_path + str(text_file) + '/TumorPixels/tumor_' + str(idx) + '.txt', 'r')
        tumor_line = f1.readlines()
        tumor_lines.append(tumor_line)
        tumor_id.append(len(tumor_line))
        f1.close()

        f2 = open(data_path + str(text_file) + '/LiverPixels/liver_' + str(idx) + '.txt', 'r')
        liver_line = f2.readlines()
        liver_lines.append(liver_line)
        liver_id.append(len(liver_line))
        f2.close()

    time_stamp_2 = time.time()

    print(time_stamp_2 - time_stamp_1)

    return data_id, volume_list, segmentation_list, tumor_lines, liver_lines, tumor_id, liver_id, min_index_list, max_index_list


def train_and_predict():
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model = DenseUNet(reduction=0.5, weights_path='./result_train_dense167_fast/model/weights365.04-0.02.hdf5')
    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=[weighted_cross_entropy])

    data_id, volume_list, segmentation_list, tumor_lines, liver_lines, tumor_id, liver_id, min_index_list, max_index_list = load_fast_files()

    # print (model.summary())
    if not os.path.exists(path + "model"):
        os.mkdir(path + 'model')
        os.mkdir(path + 'history')
    else:
        if os.path.exists(path + "history/loss_batch.txt"):
            os.remove(path + 'history/loss_batch.txt')
        if os.path.exists(path + "history/loss_epoch.txt"):
            os.remove(path + 'history/loss_epoch.txt')
    model_checkpoint = ModelCheckpoint(path + 'model/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=1,
                                       save_best_only=False, save_weights_only=False, mode='min', period=2)

    print('-' * 30)
    print('Fitting model......')
    print('-' * 30)

    steps = 27386 / batch_size
    model.fit_generator(generate_arrays_from_file(data_id, volume_list, segmentation_list, tumor_lines, liver_lines, tumor_id, liver_id, min_index_list, max_index_list),
                        steps_per_epoch=steps, epochs=6000, verbose=1, callbacks=[model_checkpoint], max_queue_size=10, workers=3, use_multiprocessing=True)

    print('Finished Training .......')


if __name__ == '__main__':
    train_and_predict()