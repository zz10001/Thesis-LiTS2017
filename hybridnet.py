import tensorflow as tf
from keras.models import Model
from keras.layers import Input, ZeroPadding2D, concatenate, Lambda, ZeroPadding3D, add
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv3D, UpSampling3D, AveragePooling3D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from lib.custom_layers import Scale


concat_axis = 3


def DenseUNet(img_input, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.0, dropout_rate=0.0):
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
    # From architecture for ImageNet (Table 1 in the paper)
    nb_layers = [6, 12, 36, 24]
    box = []
    stage = 0

    x = ZeroPadding2D((3, 3), name='convolution_1_zero_padding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='convolution_1', use_bias=False, trainable=True)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, momentum=1, name='convolution_1_batch_normalization', trainable=False)(x, training=False)
    x = Scale(axis=concat_axis, name='convolution_1_scale')(x)
    x = Activation('relu', name='convolution_1_ReLU')(x)

    box.append(x)

    x = ZeroPadding2D((1, 1), name='pooling_1_zero_padding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pooling_1')(x)

    for block_id in range(nb_dense_block - 1):
        stage = block_id + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_id], nb_filter, growth_rate, dropout_rate=dropout_rate)

        box.append(x)

        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, momentum=1, name='convolution_'+str(final_stage)+'_blk_batch_normalization', trainable=False)(x, training=False)
    x = Scale(axis=concat_axis, name='convolution_'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='ReLU_'+str(final_stage)+'_blk')(x)
    box.append(x)

    up0 = UpSampling2D(size=(2, 2))(x)
    conv_up0 = Conv2D(768, (3, 3), padding="same", name="upsampling_0", trainable=True)(up0)
    bn_up0 = BatchNormalization(name="batch_normalization_upsampling_0", momentum=1, trainable=False)(conv_up0, training=False)
    ac_up0 = Activation('relu', name='ReLU_upsampling_0')(bn_up0)

    up1 = UpSampling2D(size=(2, 2))(ac_up0)
    conv_up1 = Conv2D(384, (3, 3), padding="same", name="upsampling_1", trainable=True)(up1)
    bn_up1 = BatchNormalization(name="batch_normalization_upsampling_1", momentum=1, trainable=False)(conv_up1, training=False)
    ac_up1 = Activation('relu', name='ReLU_upsampling_1')(bn_up1)

    up2 = UpSampling2D(size=(2, 2))(ac_up1)
    conv_up2 = Conv2D(96, (3, 3), padding="same", name="upsampling_2", trainable=True)(up2)
    bn_up2 = BatchNormalization(name="batch_normalization_upsampling_2", momentum=1, trainable=False)(conv_up2, training=False)
    ac_up2 = Activation('relu', name='ReLU_upsampling_2')(bn_up2)

    up3 = UpSampling2D(size=(2, 2))(ac_up2)
    conv_up3 = Conv2D(96, (3, 3), padding="same", name="upsampling_3", trainable=True)(up3)
    bn_up3 = BatchNormalization(name="batch_normalization_upsampling_3", momentum=1, trainable=False)(conv_up3, training=False)
    ac_up3 = Activation('relu', name='ReLU_upsampling_3')(bn_up3)

    up4 = UpSampling2D(size=(2, 2))(ac_up3)
    conv_up4 = Conv2D(64, (3, 3), padding="same", name="upsampling_4", trainable=True)(up4)
    bn_up4 = BatchNormalization(name="batch_normalization_upsampling_4", momentum=1, trainable=False)(conv_up4, training=False)
    ac_up4 = Activation('relu', name='ReLU_upsampling_4')(bn_up4)

    x = Conv2D(3, (1, 1), padding="same", name='2D-DenseUNet-167_classifier', trainable=True)(ac_up4)

    return ac_up4, x


def DenseNet3D(img_input, nb_dense_block=4, growth_rate=32, nb_filter=96, reduction=0.0, dropout_rate=0.0):
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

    # From architecture for ImageNet (Table 1 in the paper)
    nb_layers = [3, 4, 12, 8]
    box = []
    stage = 0

    x = ZeroPadding3D((3, 3, 3), name='3D_convolution_1_zero_padding')(img_input)
    x = Conv3D(nb_filter, (7, 7, 7), strides=(2, 2, 2), name='3D_convolution_1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=4, name='3D_convolution_1_batch_normalization')(x)
    x = Scale(axis=4, name='3D_convolution_1_scale')(x)
    x = Activation('relu', name='3D_convolution_1_ReLU')(x)

    box.append(x)

    x = ZeroPadding3D((1, 1, 1), name='3D_pooling_1_zero_padding')(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), name='3D_pooling_1')(x)

    for block_id in range(nb_dense_block - 1):
        stage = block_id + 2
        x, nb_filter = dense_block3d(x, stage, nb_layers[block_id], nb_filter, growth_rate, dropout_rate=dropout_rate)
        box.append(x)
        # Add transition_block
        x = transition_block3d(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block3d(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate)

    x = BatchNormalization(epsilon=eps, axis=4, name='3D_convolution_' + str(final_stage) + '_blk_batch_normalization')(x)
    x = Scale(axis=4, name='3D_convolution_' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='3D_convolution_ReLU' + str(final_stage) + '_blk')(x)

    box.append(x)

    up0 = UpSampling3D(size=(2, 2, 1))(x)
    conv_up0 = Conv3D(504, (3, 3, 3), padding="same", name="3D_upsampling_0")(up0)
    bn_up0 = BatchNormalization(name="3D_upsampling_0_batch_normalization")(conv_up0)
    ac_up0 = Activation('relu', name='3D_upsampling_0_ReLU')(bn_up0)

    up1 = UpSampling3D(size=(2, 2, 1))(ac_up0)
    conv_up1 = Conv3D(224, (3, 3, 3), padding="same", name="3D_upsampling_1")(up1)
    bn_up1 = BatchNormalization(name="3D_upsampling_1_batch_normalization")(conv_up1)
    ac_up1 = Activation('relu', name='3D_upsampling_1_ReLU')(bn_up1)

    up2 = UpSampling3D(size=(2, 2, 1))(ac_up1)
    conv_up2 = Conv3D(192, (3, 3, 3), padding="same", name="3D_upsampling_2")(up2)
    bn_up2 = BatchNormalization(name="3D_upsampling_2_batch_normalization")(conv_up2)
    ac_up2 = Activation('relu', name='3D_upsampling_2_ReLU')(bn_up2)

    up3 = UpSampling3D(size=(2, 2, 2))(ac_up2)
    conv_up3 = Conv3D(96, (3, 3, 3), padding="same", name="3D_upsampling_3")(up3)
    bn_up3 = BatchNormalization(name="3D_upsampling_3_batch_normalization")(conv_up3)
    ac_up3 = Activation('relu', name='3D_upsampling_3_ReLU')(bn_up3)

    up4 = UpSampling3D(size=(2, 2, 2))(ac_up3)
    conv_up4 = Conv3D(64, (3, 3, 3), padding="same", name="3D_upsampling_4")(up4)
    bn_up4 = BatchNormalization(name="3D_upsampling_4_batch_normalization")(conv_up4)
    ac_up4 = Activation('relu', name='3D_upsampling_4_ReLU')(bn_up4)

    x = Conv3D(3, (1, 1, 1), padding="same", name='3D_classifier')(ac_up4)

    return ac_up4, x


def conv_block3d(x, stage, branch, nb_filter, dropout_rate=None):
    """Apply BatchNorm, Relu, bottleneck 1x1 Conv3D, 3x3 Conv3D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    """
    eps = 1.1e-5
    conv_name_base = '3D_convolution_' + str(stage) + '_' + str(branch)
    relu_name_base = '3D_ReLU_' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=4, name=conv_name_base+'_x1_batch_normalization', momentum=1.0, trainable=False)(x, training=False)
    x = Scale(axis=4, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv3D(inter_channel, (1, 1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = BatchNormalization(epsilon=eps, axis=4, name=conv_name_base+'_x2_batch_normalization', momentum=1.0, trainable=False)(x, training=False)
    x = Scale(axis=4, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding3D((1, 1, 1), name=conv_name_base+'_x2_zero_padding')(x)
    x = Conv3D(nb_filter, (3, 3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def dense_block3d(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True):
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
        x = conv_block3d(concat_feat, stage, branch, growth_rate, dropout_rate)
        concat_feat = concatenate([concat_feat, x], axis=4, name='3d_concatenation_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def transition_block3d(x, stage, nb_filter, compression=1.0, dropout_rate=None):
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
    conv_name_base = '3D_convolution_' + str(stage) + '_blk'
    relu_name_base = '3D_ReLU_' + str(stage) + '_blk'
    pool_name_base = '3D_pooling_' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=4, name=conv_name_base+'_batch_normalization', momentum=1.0)(x, training=False)
    x = Scale(axis=4, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv3D(int(nb_filter * compression), (1, 1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling3D((2, 2, 1), strides=(2, 2, 1), name=pool_name_base)(x)

    return x


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

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, momentum=1, name=conv_name_base + '_x1_batch_normalization', trainable=False)(x, training=False)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base + '_x1', use_bias=False, trainable=True)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, momentum=1, name=conv_name_base + '_x2_batch_normalization', trainable=False)(x, training=False)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zero_padding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base + '_x2', use_bias=False, trainable=True)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None):
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
    conv_name_base = 'convolution_' + str(stage) + '_blk'
    relu_name_base = 'ReLU_' + str(stage) + '_blk'
    pool_name_base = 'pooling_' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, momentum=1, name=conv_name_base+'_batch_normalization', trainable=False)(x, training=False)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False, trainable=True)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True):
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
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concatenation_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def slice_column_axis(x, h1, h2):
    # Slice from h1 to h2 index of the column axis of the x matrix
    return x[:, :, :, h1:h2, :]


def slice2d(x, h1, h2):
    # Slice, transpose then add another dimension to the x matrix
    tmp = x[h1:h2, :, :, :]
    tmp = tf.transpose(tmp, perm=[1, 2, 0, 3])
    tmp = tf.expand_dims(tmp, 0)
    return tmp


def slice_last(x):
    # Slice out the last layer
    x = x[:, :, :, :, 0]
    return x


def dense_rnn_net(args):

    #  ************************3d volume input******************************************************************
    img_input = Input(batch_shape=(args.batch, args.input_size, args.input_size, args.input_cols, 1), name='volumetric_data')

    #  ************************(batch*d3cols)*2d-volume--2D DenseNet branch**************************************
    input2d = Lambda(slice_column_axis, arguments={'h1': 0, 'h2': 2})(img_input)
    single = Lambda(slice_column_axis, arguments={'h1': 0, 'h2': 1})(img_input)
    input2d = concatenate([single, input2d], axis=3)
    for i in range(args.input_cols - 2):
        input2d_tmp = Lambda(slice_column_axis, arguments={'h1': i, 'h2': i + 3})(img_input)
        input2d = concatenate([input2d, input2d_tmp], axis=0)
        if i == args.input_cols - 3:
            final1 = Lambda(slice_column_axis, arguments={'h1': args.input_cols - 2, 'h2': args.input_cols})(img_input)
            final2 = Lambda(slice_column_axis, arguments={'h1': args.input_cols - 1, 'h2': args.input_cols})(img_input)
            final = concatenate([final1, final2], axis=3)
            input2d = concatenate([input2d, final], axis=0)
    input2d = Lambda(slice_last)(input2d)

    #  ******************************stack to 3D volumes *******************************************************
    feature2d, classifier2d = DenseUNet(input2d, reduction=0.5)
    res2d = Lambda(slice2d, arguments={'h1': 0, 'h2': 1})(classifier2d)
    fea2d = Lambda(slice2d, arguments={'h1': 0, 'h2': 1})(feature2d)
    for j in range(args.input_cols - 1):
        score = Lambda(slice2d, arguments={'h1': j + 1, 'h2': j + 2})(classifier2d)
        fea2d_slice = Lambda(slice2d, arguments={'h1': j + 1, 'h2': j + 2})(feature2d)
        res2d = concatenate([res2d, score], axis=3)
        fea2d = concatenate([fea2d, fea2d_slice], axis=3)

    #  *************************** 3d DenseNet on 3D volume (concatenate with feature map )*********************************
    res2d_input = Lambda(lambda x: x * 250)(res2d)
    input3d_ori = Lambda(slice_column_axis, arguments={'h1': 0, 'h2': args.input_cols})(img_input)
    input3d = concatenate([input3d_ori, res2d_input], axis=4)
    feature3d, classifier3d = DenseNet3D(input3d, reduction=0.5)

    final = add([feature3d, fea2d])
    final_convolution = Conv3D(64, (3, 3, 3), padding="same", name='final_convolution')(final)
    final_convolution = Dropout(rate=0.3)(final_convolution)
    final_bn = BatchNormalization(name="final_batch_normalization")(final_convolution)
    final_ac = Activation('relu', name='final_ReLU')(final_bn)
    classifier = Conv3D(3, (1, 1, 1), padding="same", name='2d_3d_classifier')(final_ac)

    model = Model(inputs=img_input, outputs=classifier, name='auto3d_residual_convolution')

    return model
