""" Convolutional Neural Network.
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import read_data
import numpy as np

# prepare data at first
read_data.prepare_data()

# need to restore/save model ?
NEED_RESTORE = False
NEED_SAVE = True


LEFT_PATH = "E:\\FashionAI_Data\\fashionAI_point\\"

# 模型保存的路径，模型可以保存然后下次读取出来继续训练
model_path = LEFT_PATH + "\\tmp\\"+read_data.CLOTHES_TYPE+"model_3level.ckpt"

# 训练结果保存的路径
SAVE_PATH = "E:\\FashionAI_Data\\fashionAI_point\\fashionAI_key_points_test_a_20180227" \
            "\\test\\Images\\"+read_data.CLOTHES_TYPE+"_res.csv"


# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 4
display_step = 200

image_size = read_data.IMAGE_SIZE
# Network Parameters
y_dimen = read_data.y_dimen
dropout = 0.8 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, image_size,image_size,3])
Y = tf.placeholder(tf.float32, [None, y_dimen])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.tanh(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    #x = tf.reshape(x, shape=[-1, image_size, image_size, 3])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides=2)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=4)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.tanh(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([8, 8, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5,5, 32, 64])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([4, 4, 64, 128])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8*8*128, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, y_dimen]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([y_dimen]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
logit1 = logits[0]

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.square(logits-Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Restore model weights from previously saved model
    if NEED_RESTORE:
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)

    for step in range(1, num_steps+1):
        batch_x, batch_y = read_data.next_train_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and coordinates
            logit_1,loss= sess.run([logits, loss_op], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + "\n Minibatch Loss= " + \
                  "{:.0f}".format(loss))
            for lo in logit_1:
                print("{:.0f},{:.0f}".format(lo[0], lo[1]))
            print("------------------------------")

    # Save trained model
    if NEED_SAVE:
        save_path = saver.save(sess, model_path)
        print("Model saved from file: %s" % model_path)

    # Predict coordinates
    res = sess.run(logits, feed_dict={X: read_data.test_data_x,
                                Y: read_data.test_data_y,
                                keep_prob: 1.0})

    # Write result into file
    res = np.asarray(res)
    res = res.astype(int)
    with open(SAVE_PATH, "w") as f:
        for line in res:
            clo_idx = 0
            line_to_write = ""
            for i,od in enumerate(line):
                while clo_idx < len(read_data.y_valid) and read_data.y_valid[clo_idx]==False:
                    line_to_write = line_to_write+"-1_-1_-1,"
                    clo_idx = clo_idx + 1
                if i % 2 == 0:
                    line_to_write = line_to_write + str(od)
                else:
                    line_to_write = line_to_write + "_{:.0f}_1,".format(od)
                    clo_idx = clo_idx + 1

            while clo_idx < len(read_data.y_valid) and read_data.y_valid[clo_idx] == False:
                line_to_write = line_to_write + "-1_-1_-1,"
                clo_idx = clo_idx + 1
            line_to_write.strip(",")
            f.write(line_to_write+"\n")

    print("Optimization Finished!")