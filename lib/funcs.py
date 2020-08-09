import numpy as np
from keras import backend as k


def predict_tumor_in_window(model, img_test, num, mini, maxi, args):
    batch = args.batch
    img_depth = args.input_size
    img_row = args.input_size
    img_column = args.input_cols

    window_cols = (img_column/4)
    count = 0
    box_test = np.zeros((batch, img_depth, img_row, img_column, 1), dtype="float32")

    x = img_test.shape[0]
    y = img_test.shape[1]
    z = img_test.shape[2]

    right_cols = int(min(z, maxi[2]+10)-img_column)
    left_cols = max(0, min(mini[2]-5, right_cols))
    score = np.zeros((x, y, z, num), dtype='float32')
    score_num = np.zeros((x, y, z, num), dtype='int16')

    for cols in range(left_cols, right_cols+window_cols, window_cols):
        if cols > z - img_column:
            patch_test = img_test[0:img_depth, 0:img_row, z - img_column:z]
            box_test[count, :, :, :, 0] = patch_test
            patch_test_mask = model.predict(box_test, batch_size=batch, verbose=0)
            patch_test_mask = k.softmax(patch_test_mask)
            patch_test_mask = k.eval(patch_test_mask)
            patch_test_mask = patch_test_mask[:, :, :, 1:-1, :]

            for i in range(batch):
                score[0:img_depth, 0:img_row,  z-img_column+1:z-1, :] += patch_test_mask[i]
                score_num[0:img_depth, 0:img_row,  z-img_column+1:z-1, :] += 1
        else:
            patch_test = img_test[0:img_depth, 0:img_row, cols:cols + img_column]
            box_test[count, :, :, :, 0] = patch_test
            patch_test_mask = model.predict(box_test, batch_size=batch, verbose=0)
            patch_test_mask = k.softmax(patch_test_mask)
            patch_test_mask = k.eval(patch_test_mask)
            patch_test_mask = patch_test_mask[:, :, :, 1:-1, :]
            for i in range(batch):
                score[0:img_depth, 0:img_row, cols+1:cols+img_column-1, :] += patch_test_mask[i]
                score_num[0:img_depth, 0:img_row, cols+1:cols+img_column-1, :] += 1

    score = score/(score_num+1e-4)
    score1 = score[:, :, :, num-2]
    score2 = score[:, :, :, num-1]

    return score1, score2
