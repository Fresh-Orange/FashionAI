# coding=utf-8
from PIL import Image
import numpy as np
import pandas as pd

LEFT_PATH = "E:\\FashionAI_Data\\fashionAI_point\warm_up_train_20180222\\train\\"
ANNOTATION_PATH = LEFT_PATH+"Annotations\\"
BLOUSE_PATH = LEFT_PATH+"\\Annotations\\blouse.csv"
TEST_PATH = LEFT_PATH+"\\Annotations\\blouse_test.csv"
RATIO = 0.6
IMAGE_SIZE = 32
y_idx = np.arange(2,8+1)
y_idx = np.hstack((y_idx, np.arange(11,17)))
y_dimen = len(y_idx)*2

train_data_x = []
train_data_y = []
validation_data_x = []
validation_data_y = []
test_data_x = []
test_data_y = []

is_prepared = False

def prepare_data():
    global train_data_x,train_data_y,validation_data_x,\
        validation_data_y,is_prepared,test_data_x,test_data_y,y_idx

    # pandas read csv
    df = pd.read_csv(BLOUSE_PATH)
    image_paths = df.ix[:, 'image_id']
    neckline_left = df.ix[:, y_idx]
    neckline_left = np.asarray(neckline_left)

    # compute number of train data
    train_num = round(image_paths.size * RATIO)

    # read X data(which is image)
    for i,p in enumerate(image_paths):
        im = Image.open(LEFT_PATH + p)
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        if i < train_num:
            train_data_x.append(np.asarray(im))
        else:
            validation_data_x.append(np.asarray(im))

    # turn list data to ndarray
    train_data_x = (np.asarray(train_data_x) / 255) - 0.5
    validation_data_x = (np.asarray(validation_data_x) / 255) - 0.5

    #print(train_data_x.shape)

    temp_y = []
    # read Y data(which is coordinate)
    for yi in range(len(y_idx)):
        split_neckline_left = [x.split("_") for x in neckline_left[:,yi]]
        y = np.asarray(split_neckline_left, 'int32')
        y = np.reshape(y, [-1, 3])
        if yi == 0:
            temp_y = y[:, 0:2]
        else:
            temp_y = np.hstack((temp_y, y[:, 0:2]))

    train_data_y = temp_y[0:train_num]
    validation_data_y = temp_y[train_num:]

    # print(train_data_y.shape)
    # print(train_data_y[0])

    ############################### READ TEST DATA ###############################
    # pandas read csv
    df = pd.read_csv(TEST_PATH)
    image_paths = df.ix[:, 'image_id']
    neckline_left = df.ix[:, y_idx]
    neckline_left = np.asarray(neckline_left)


    # read X data(which is image)
    for i, p in enumerate(image_paths):
        im = Image.open(LEFT_PATH + p)
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        test_data_x.append(np.asarray(im))

    # turn list data to ndarray
    test_data_x = (np.asarray(test_data_x) / 255) - 0.5

    # read Y data(which is coordinate)
    for yi in range(len(y_idx)):
        split_neckline_left = [x.split("_") for x in neckline_left[:,yi]]
        y = np.asarray(split_neckline_left, 'int32')
        y = np.reshape(y, [-1, 3])
        if yi == 0:
            test_data_y = y[:, 0:2]
        else:
            test_data_y = np.hstack((test_data_y, y[:, 0:2]))

    print("prepared!")
    is_prepared = True


def next_batch(batch_size, data, labels):
    """
    Return a total of `num` random samples and labels.
    """
    #print(is_prepared)
    if not is_prepared:
        raise ValueError("data is not prepared")
    if len(data) != len(labels):
        raise ValueError("lengths unequal")
    if batch_size > len(data):
        raise ValueError("num of samples too big")

    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def next_train_batch(batch_size):
    return next_batch(batch_size, train_data_x, train_data_y)

# def next_validation_batch(batch_size):
#     return next_batch(batch_size, validation_data_x, validation_data_y)

def get_validation_x():
    return validation_data_x

def get_validation_y():
    return validation_data_y

def get_test_x():
    return test_data_x

def get_test_y():
    return test_data_y

#prepare_data()