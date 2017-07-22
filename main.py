import tensorflow as tf
import csv
import pickle
import glob, os
import numpy as np

scan_file = "scan_data.csv"
scan_path = "/media/ADANN_DATA/data"
# sess = tf.InteractiveSession()

x = tf.placeholder('float')
y = tf.placeholder('float')

BATCH_SIZE = 1
DROPOUT_RATE = .2
VALIDATION_SIZE = 100

def get_scan(key):
    print("Loading scan " + key)
    os.chdir(scan_path)
    name = glob.glob("*" + key + "*.nii")[0]
    with open(name, mode='rb') as f:
        return pickle.loads(f.read())


def conv3d(x, window):
    return tf.nn.conv3d(x, window, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def get_next_batch(batch_size, batch_number, train_data):
    index = batch_size * batch_number
    return dict(list(train_data.items())[index:batch_size])

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,5,1,5])),
               'W_fc':tf.Variable(tf.random_normal([12345,25])),
               'out':tf.Variable(tf.random_normal([25, 3]))}
    biases = {'b_conv1':tf.Variable(tf.random_normal([5])),
              'b_fc':tf.Variable(tf.random_normal([25])),
              'out':tf.Variable(tf.random_normal([3]))}
    x = tf.reshape(x, shape=[-1,32,32,20,1])
    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)
    fc = tf.reshape(conv1, [-1, 12345])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, 1.0-DROPOUT_RATE)
    output = tf.matmul(fc, weights['out']) + biases['out']
    return output

def train_neural_network(x, train_data, validation_data):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    epochs_to_run = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        batches_to_run = 10
        for epoch in range(epochs_to_run):
            epoch_loss = 0
            for batch_number in range(batches_to_run):
                data = get_next_batch(BATCH_SIZE, batch_number, train_data)
                for scan_id in data:
                    try:
                        X = get_scan(scan_id)
                        Y = data[scan_id]
                        _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                        epoch_loss += c
                    except Exception as e:
                        pass
            print('Epoch', epoch+1, 'completed out of', epochs_to_run, 'loss:', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            
            print('Accuracy:', accuracy.eval({ x:[get_scan(i) for i in validation_data], y: validation_data.values() }))
             

def main():
    needed_data = {}
    with open(scan_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0] != '':
                needed_row = []
                # needed_row.append(row[1])
                needed_row.append(row[3])
                # needed_data.append(needed_row)
                needed_data[row[1]] = needed_row 
    training_data = dict(list(needed_data.items())[:-100])
    validation_data = dict(list(needed_data.items())[-10:])
    train_neural_network(x, training_data, validation_data)

if __name__ == "__main__":
    main()

