import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy

def neural_loss(xtra):
    train_X = np.asarray(xtra)
    train_X = np.reshape(train_X, (1, 1, 2))
    train_Y = np.asarray(xtra)
    train_Y = np.reshape(train_Y, (1, 2))
    X = tf.placeholder(tf.float32, [None, 1, 2])
    W1 = tf.Variable(tf.truncated_normal([2, 100], stddev=0.1))
    # W1=tf.Variable(tf.truncated_normal([3,3],stddev=0.1))
    # b1=tf.Variable(tf.zeros([3]))
    b1 = tf.Variable(tf.zeros([100]))
    W2 = tf.Variable(tf.truncated_normal([100, 100], stddev=0.1))
    b2 = tf.Variable(tf.zeros([100]))
    W3 = tf.Variable(tf.truncated_normal([100, 3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([3]))

    init = tf.global_variables_initializer()

    # model

    Y1 = (tf.matmul(tf.reshape(X, [-1, 2]), W1) + b1)
    Y2 = (tf.matmul(Y1, W2) + b2)
    Y3 = (tf.matmul(Y2, W3) + b3)

    YF1 = l1 * tf.cos(Y3[:, 0]) + l2 * tf.cos(Y3[:, 0] + Y3[:, 1]) + l3 * tf.cos(Y3[:, 0] + Y3[:, 1] + Y3[:, 2])
    YF2 = l1 * tf.sin(Y3[:, 0]) + l2 * tf.sin(Y3[:, 0] + Y3[:, 1]) + l3 * tf.sin(Y3[:, 0] + Y3[:, 1] + Y3[:, 2])
    Y = tf.stack([YF1, YF2], axis=1)
    # Y=tf.nn.softmax(tf.matmul(tf.reshape(X,[-1,3]),W1)+b1)
    # placeholder for correct answers
    Y_ = tf.placeholder(tf.float32, [None, 2])
    print(Y)
    # loss function
    # cross_entropy=-tf.reduce_sum(Y_*tf.log(Y))
    cross_entropy = tf.reduce_sum(tf.square(Y - Y_))
    # cross_entropy=tf.losses.mean_squared_error(Y, Y_)


    optimizer = tf.train.GradientDescentOptimizer(0.003)
    train_step = optimizer.minimize(cross_entropy)

    sess = tf.Session()
    sess.run(init)


    for i in range(800):
        train_data = {X: train_X, Y_: train_Y}
        # _,c1=sess.run([train_step, cross_entropy], feed_dict=train_data)
        sess.run(train_step, feed_dict=train_data)
        c1 = sess.run([cross_entropy], feed_dict=train_data)
        if i==799:
            ctrain.append(c1)
            #print(sess.run(X,feed_dict=train_data))
            #print(sess.run(Y3, feed_dict=train_data))
            print(c1)

    return(ctrain)

ctrain = []
l1=1
l2=1
l3=1
q1=3.1415*np.random.rand(500)
q1=np.asarray(q1)
q1=np.reshape(q1,(500,1))
q2=-3.1415*np.random.rand(500)
q2=np.asarray(q2)
q2=np.reshape(q2,(500,1))
t1=3.1415*np.random.rand(250)/2
t2=-3.1415*np.random.rand(250)/2
q3=np.vstack((t1,t2))
q3=np.asarray(q3)
q3=np.reshape(q3,(500,1))
train_Y=np.column_stack((q1,q2,q3))
xe=l1*np.cos(q1)+l2*np.cos(q1+q2)+l3*np.cos(q1+q2+q3)
ye=l1*np.sin(q1)+l2*np.sin(q1+q2)+l3*np.sin(q1+q2+q3)
xtr=np.column_stack((xe,ye))
print(xtr.shape)

for j in range(500):
    ctrain=neural_loss(xtr[j])
    print(xtr[j])


print(ctrain)
plt.plot(ctrain, 'ro')
plt.ylabel('loss')
plt.xlabel('number of example')
plt.show()