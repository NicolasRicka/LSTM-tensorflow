''' 
This file contains the single layered LSTM RNN, implemented in tensorflow,
Together with a training loop.

This is a preliminary work for the Kaggle competition 'predict future sales'.
'''


import random as rdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



''' Creating a phony dataset to test the LSTM network '''
datasize = 2500

datanp = np.array([[ np.sin(i * np.pi / 100) for i in range(datasize)] for j in range(5)])

#Add some noise to see the effect on prediction
datanp += 0.2* np.random.random(datanp.shape)

plt.plot(datanp[0], 'ro')




#loading hyperparameters
#from text_prediction import text_prediction
from NicoHyperparameters import hyperparameters 
ratio_test,seq_size,num_hidden,batch_size, epoch = hyperparameters


#loading data
#datanp = np.load('data_np.npy')
data_size = datanp.shape[1]
data_dim = datanp[0,0].size

test_size = int(ratio_test * data_size)
train_size = data_size - test_size


train_input = datanp[:,:train_size]
test_input = datanp[:,train_size - data_size-seq_size:]
print('test input and output created')


'''Construction of the LSTMN'''
#Creating the computation graph
tf.reset_default_graph()

data = tf.placeholder(tf.float32, [None, seq_size, data_dim])
target = tf.placeholder(tf.float32, [None, data_dim])

cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple = True)

val, state = tf.nn.dynamic_rnn(cell, data, dtype = tf.float32)
val = tf.transpose(val, [1,0,2])
last = tf.gather(val, int(val.get_shape()[0])-1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.matmul(last, weight) + bias
squared_error = tf.reduce_mean((target - prediction)**2)

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(squared_error)

saver = tf.train.Saver()

#Execution of tensorflow computation graph
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

tf.summary.tensor_summary('squared_error', squared_error)
tf.summary.tensor_summary('error_on_test_set', squared_error)
merged_summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter('./graphs', sess.graph)
if batch_size != 0:
    no_of_batches = int((train_size-seq_size-1)/batch_size)
    #no_of_batches = int(len(train_input)/batch_size)
else:
    batch_size = train_size-seq_size-1
    #batch_size = len(train_input)
    no_of_batches = 1

indices = list(range(train_size - seq_size))
rdm.shuffle(indices)
for i in range(epoch):
    
    for seq_index in range(train_input.shape[0]):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = np.array([  train_input[seq_index,i:i+seq_size] for i in indices[ptr:ptr+batch_size]] ).reshape([-1,seq_size, data_dim]), np.array([  train_input[seq_index,i+seq_size] for i in indices[ptr:ptr+batch_size]] ).reshape([-1, data_dim])
            #inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
            ptr+=batch_size
            sess.run(minimize,{data: inp, target: out})
            #print("Epoch - "+str(i) + " batch - " + str(j) + "/" + str(nb_of_batches))
    print("Epoch - "+str(i))
#    print(text_prediction(sess.run(prediction,{data: 'k'})))
    cost = 0
    for seq_index in range(datanp.shape[0]):
        inp = np.array([  test_input[seq_index,i:i+seq_size] for i in range(test_size-seq_size) ] ).reshape([-1,seq_size, data_dim])
        out = np.array([  test_input[seq_index,i+seq_size] for i in range(test_size-seq_size)]).reshape([-1, data_dim])
        
        pre, sqer = sess.run([prediction,squared_error],{data: inp, target: out})
        cost += sqer
        #print(cost, pre, out)
    curr_cost = sess.run(squared_error,{data: inp, target: out})
    
    #curr_cost,summary = sess.run([squared_error,merged_summary_op],{data: inp, target: out})
    print('Epoch {:2d} train cost {:3.3f} test cost {}'.format(i + 1, cost, curr_cost))
    #writer.add_summary(summary, i)
    generated_data = test_input[0,0:seq_size].reshape([data_dim, seq_size, -1])
    current_data = generated_data
    for i in range(test_size):
            onehot_pred = sess.run(prediction, feed_dict={data: current_data})
            current_data = current_data[:,1:,:]
            current_data = np.append(current_data, np.array([onehot_pred ]).reshape([1,1,-1]),axis = 1)
            generated_data = np.append(generated_data, np.array([onehot_pred]).reshape([1,1,-1]),axis = 1)
    plt.plot(generated_data.reshape([-1]), 'rx')
    plt.plot(datanp[0,train_size - data_size-seq_size:])
    plt.show()
weight_value = sess.run(weight)
bias_value = sess.run(bias)

save_path = saver.save(sess, "/tmp/model.ckpt")
print("Model saved in path: %s" % save_path)
sess.close()




with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("weight_value : %s" % weight.eval())
  print("bias_value : %s" % bias.eval())
  pred = sess.run(prediction,{data: test_input[0,0:seq_size].reshape([data_dim, seq_size, -1])})
  print('prediction:', pred)
  



def pred_fun(length_of_story, starting_data):
    '''Use the LSTMN stored in model.cpkt to generate a prediction from previous data'''
    generated_data = starting_data.copy()
    print(generated_data.shape)
    generated_data.reshape([1, seq_size, -1])
    current_data = generated_data
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
        print('Hyperparameters',hyperparameters)
        for i in range(length_of_story):
            onehot_pred = sess.run(prediction, feed_dict={data: current_data})
            current_data = current_data[:,1:,:]
            current_data = np.append(current_data, np.array([onehot_pred]).reshape([1,1,-1]),axis = 1)
            generated_data = np.append(generated_data, np.array([onehot_pred]).reshape([1,1,-1]),axis = 1)
    return generated_data
    #except:
    #    print("Word not in dictionary")


generated_data = pred_fun(200, test_input[0,0:seq_size].reshape([data_dim, seq_size, -1]))
print(generated_data)


