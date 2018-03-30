# coding=utf-8
from PIL import Image
import numpy as np
import pandas as pd
types = ["blouse", "skirt","outwear","dress","trousers"]
CLOTHES_TYPE = types[3]
TRAIN_ROOT_PATH = "E:\\FashionAI_Data\\fashionAI_point\\fashionAI_key_points_train_20180227\\train\\"
TEST_ROOT_PATH = "E:\\FashionAI_Data\\fashionAI_point\\fashionAI_key_points_test_a_20180227\\test\\"
ANNOTATION_PATH = TRAIN_ROOT_PATH + "Annotations\\"
TRAIN_PATH = TRAIN_ROOT_PATH + "Annotations\\train.csv"
TEST_PATH = "E:\\FashionAI_Data\\fashionAI_point\\fashionAI_key_points_test_a_20180227\\test\\test.csv"
RATIO = 0.95
IMAGE_SIZE = 32

y_valid = []
y_idx = []
y_dimen = 0

train_data_x = []
train_data_y = []
validation_data_x = []
validation_data_y = []
test_data_x = []
test_data_y = []

is_prepared = False

def outlier_process(data):
    mean_row = np.mean(data, axis=0)
    col_num = data.shape[1]
    print("processing outliers, col_num = ", col_num)
    for i in range(col_num):
        data[data[:, 0] == -1, i] = mean_row[i]
    return data

def get_valid_idx(data_frame):
    """
    判定哪些列是有效列，有效列是指非全部都是“-1_-1_-1”的；列
    :param data_frame:
    :return:
    """
    global y_idx,y_dimen,y_valid
    y_valid = []
    y_idx = []
    for i in range(len(data_frame.columns[2:])):
        i = i + 2
        start = data_frame.index[0]
        y_valid.append(data_frame.ix[start, i] != data_frame.ix[start+1, i])
        if data_frame.ix[start, i] != data_frame.ix[start+1, i]:
            y_idx.append(i)
    print("valid indexs = ",y_idx)
    y_dimen = len(y_idx)*2
    return y_idx

def prepare_data():
    global train_data_x,train_data_y,validation_data_x,\
        validation_data_y,is_prepared,test_data_x,test_data_y,y_idx,y_dimen


    # pandas read csv
    df = pd.read_csv(TRAIN_PATH)
    df = df[df["image_category"] == CLOTHES_TYPE] # select clothes type
    image_paths = df.ix[:, 0]
    y_idx = get_valid_idx(df)
    opint_data = df.ix[:, y_idx]
    opint_data = np.asarray(opint_data)

    # compute number of train data
    train_num = round(image_paths.size * RATIO)


    # read X data(which is image)
    for i,p in enumerate(image_paths):
        im = Image.open(TRAIN_ROOT_PATH + p)
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        if i < train_num:
            train_data_x.append(np.asarray(im))
        else:
            validation_data_x.append(np.asarray(im))

    # zero-mean normalization and turn list data to ndarray
    train_data_x = (np.asarray(train_data_x) / 255) - 0.5
    validation_data_x = (np.asarray(validation_data_x) / 255) - 0.5

    #print(train_data_x.shape)

    temp_y = []
    # read Y data(which is coordinate)
    for yi in range(len(y_idx)):
        split_neckline_left = [x.split("_") for x in opint_data[:,yi]]
        y = np.asarray(split_neckline_left, 'int32')
        y = np.reshape(y, [-1, 3])
        if yi == 0:
            temp_y = y[:, 0:2]
        else:
            temp_y = np.hstack((temp_y, y[:, 0:2]))

    train_data_y = temp_y[0:train_num]
    train_data_y = outlier_process(train_data_y)
    validation_data_y = temp_y[train_num:]
    validation_data_y = outlier_process(validation_data_y)

    # print(train_data_y.shape)
    # print(train_data_y[0])

    ############################### READ TEST DATA ###############################
    # pandas read csv
    df = pd.read_csv(TEST_PATH)
    df = df[df["image_category"] == CLOTHES_TYPE]
    image_paths = df.ix[:,0]


    # read X data(which is image)
    for i, p in enumerate(image_paths):
        im = Image.open(TEST_ROOT_PATH + p)
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        test_data_x.append(np.asarray(im))

    # turn list data to ndarray
    test_data_x = (np.asarray(test_data_x) / 255) - 0.5


    test_data_y = np.zeros((len(test_data_x), y_dimen), dtype=int)

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