# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:28:27 2018

@author: ushasi2
link: https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
"""



import UxUyLoader as uxuy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
import scipy.io as sio
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import normalize
from numpy import linalg as LA
import stellargraph as sg
from sklearn import model_selection
#import graph_nets as gn
#import sonnet as snt

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
k = 128 # k-bits of hash code
n_classes = 100# 82#63#125
t_classes = 125
training_iters = 200
learning_rate = 0.01 
batch_size = 128
t = 300 #64 Output feature dimension
feature_dim = 256 #dimension of the pre-trained features
#n=10000 #number of pairs
alpha = 0.5

#tf.reset_default_graph()
#tf.compat.v1.reset_default_graph() 

dataset = uxuy.load_uxuy_dataset()
X = dataset[0]
Y = dataset[1]
Lx = dataset[2]
Ly = dataset[3]
nx = dataset[4]
ny = dataset[5]
W = dataset[6]
order = dataset[7]
graph = dataset[8]

Xorg = X
Yorg = Y

def onehot(code):
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(code)
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	code = onehot_encoder.fit_transform(integer_encoded)
	return code


def create_triads(X, Y, Lx, Ly, W, nx, ny, order):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''   
    triad = []
    labels = []
    cnt = 0
    x1 = np.zeros((75481,feature_dim))
    y2 = np.zeros((75481,feature_dim))
    y3 = np.zeros((75481,feature_dim))
    y1 = np.zeros((75481,feature_dim))#new
    x2 = np.zeros((75481,feature_dim))#new
    x3 = np.zeros((75481,feature_dim))#new
    w = np.zeros((75481,300))
    lx = np.zeros((75481)) 
    ly1 = np.zeros((75481)) 
    ly2 = np.zeros((75481)) 

    for xi in range(125):
    	for y in range(100):    #CHANGGGGGGGGGGGGGGGGGGEEEEEEEEEEE
        	k = np.count_nonzero(order[xi,y,:])
        	for i in range(k):
                        arr = np.arange(k)
                        np.random.shuffle(arr)
                        arr = list(arr)
                        numbers = list(range(0, 125))
                        numbers.remove(xi)			
                        x1[cnt] = X[xi,:]
			#y1[cnt] = Y[ ]#new
                        a = np.random.choice(arr)
                        b = random.choice(numbers)
                        arr.remove(i)
                        #y2_int = order[xi,:, a]
                        #y3_int = order[b,:,1]
                        p1 = np.count_nonzero(order[0:xi,0:y,0:a])
                        p2 = np.count_nonzero(order[0:b,0:y,0:1])
                        y2[cnt] = Y[p1,:]
                        y3[cnt] = Y[p2,:]
                        #x2[cnt] = Y[p1,:]#new
                        #x3[cnt] = Y[p2,:]#new
                        lx[cnt] = xi
                        ly1[cnt] = xi
                        ly2[cnt] = b
                        w[cnt] = W[xi,:]
                        cnt = cnt + 1
                        #print xi,b
                        #x1, y2, y3, w = X[x,y], order[x,y,np.random.choice(arr)], order[random.choice(numbers),y,np.random.choice(arr)], W[x,:] #np.random.choice(train[np.random.choice(arr)])
    return x1, y2, y3, w, lx, ly1, ly2


triads = create_triads(X, Y, Lx, Ly, W, nx, ny, order) ###############

X = triads[0]
Y1 = triads[1]
Y2 = triads[2]
W = triads[3]
Lx = triads[4]
Ly1 = triads[5]
Ly2 = triads[6]


#input_graphs = get_graphs()

# Create the graph network.
# convert the raw data into StellarGraph's graph format for faster operations
graph = sg.StellarGraph(nodes, edges)

generator = sg.mapper.FullBatchNodeGenerator(graph, method="gcn")

# two layers of GCN, each with hidden dimension 16
gcn = sg.layer.GCN(layer_sizes=[16, 16], generator=generator)
x_inp, x_out = gcn.in_out_tensors() # create the input and output TensorFlow tensors

# use TensorFlow Keras to add a layer to compute the (one-hot) predictions
predictions = tf.keras.layers.Dense(units=len(ground_truth_targets.columns), activation="softmax")(x_out)

# use the input and output tensors to create a TensorFlow Keras model
model = tf.keras.Model(inputs=x_inp, outputs=predictions)

'''
graph_net_module = gn.modules.GraphNetwork(
    edge_model_fn=lambda: snt.nets.MLP([32, 32]),
    node_model_fn=lambda: snt.nets.MLP([32, 32]),
    global_model_fn=lambda: snt.nets.MLP([32, 32]))
'''
# Pass the input graphs to the graph network, and return the output graphs.
#output_graphs = graph_net_module(input_graphs)

print("Training set (X) shape: {shape}", X.shape)
print("Training set (Y1) shape: {shape}", Y1.shape)
print("Training set (YLx) shape: {shape}", Lx.shape)
print("Training set (W) shape: {shape}", W.shape)

#Train
Ld = Lx
p=int(X.shape[0]*0.8) #     MAKE IT FOR ZERO-SHOT LEARNING
a = int(4*X.shape[0]/5)
train_indices = np.random.choice(a, p, replace=False)
#print min(train_indices)
#print max(train_indices)
train_X = X[train_indices]
train_Y1 = Y1[train_indices]
train_Y2 = Y2[train_indices]
train_Lx = Lx[train_indices]
train_Ly1 = Ly1[train_indices]
train_Ly2 = Ly2[train_indices]
train_Ld = Ld[train_indices]
train_Ld = train_Ld.astype(int)
#print type(train_Ld[0])
#print train_Ld
train_W = W[train_Ld ]

#Test
test_indices = np.array(list(set(range(X.shape[0])) - set(train_indices)))
test_X = X[test_indices]
test_Y1 = Y1[test_indices]
test_Y2 = Y2[test_indices]
test_Lx = Lx[test_indices]
test_Ly1 = Ly1[test_indices]
test_Ly2 = Ly2[test_indices]
test_Ld = Ld[test_indices]
test_Ld = test_Ld.astype(int)
test_W = W[test_Ld ]

#One-Hot Labels
Ld = onehot(Ld)
train_Lx = onehot(train_Lx)
train_Ly1 = onehot(train_Ly1)
train_Ly2 = onehot(train_Ly2)
test_Lx = onehot(test_Lx)
test_Ly1 = onehot(test_Ly1)
test_Ly2 = onehot(test_Ly2)



print("Training set (X) shape: {shape}", X.shape)
print("Training set (Y1) shape: {shape}", Y1.shape)
print("Training set (train_lx) shape: {shape}", train_Lx.shape)
print("Training set (test_Lx) shape: {shape}", test_Lx.shape)
print("Training set (train_ly) shape: {shape}", train_Ly1.shape)
print("Training set (test_Ly) shape: {shape}", test_Ly1.shape)
print("Training set (train_W) shape: {shape}", train_W.shape)
print("Training set (test_W) shape: {shape}", test_W.shape)



x = tf.placeholder("float", [None, feature_dim]) #0
y1 = tf.placeholder("float", [None, feature_dim]) #1
y2 = tf.placeholder("float", [None, feature_dim]) #2
lx = tf.placeholder("float", [None, n_classes]) #3
ly1 = tf.placeholder("float", [None, n_classes]) #4
ly2 = tf.placeholder("float", [None, 125]) #5
lt = tf.placeholder("float", [None, 25]) #6
l1 = tf.placeholder("float", [None, 125]) #7
w = tf.placeholder("float", [None, 300]) #8
keep_prob = tf.placeholder(tf.float32)


def load_model( sess, saver):
        latest = tf.train.latest_checkpoint(snapshot_path)
        print(latest)
        if latest == None:
            return 0
        saver.restore(sess, latest)
        i = int(latest[len(snapshot_path + 'model-'):])
        print("Model restored at %d." % i)
        return i
        
def save_model( sess, saver, i):
        if not os.path.exists(snapshot_path):
                os.makedirs(snapshot_path)
        latest = tf.train.latest_checkpoint(snapshot_path)
        if 1:
            print('Saving model at %d' % i)
            result = saver.save(sess, snapshot_path + 'model', global_step=i)
            print('Model saved to %s' % result)


weightsx = {
    'wd1x': tf.get_variable('Wx3', shape=(feature_dim,t),initializer=tf.contrib.layers.xavier_initializer()),
    'wd11x': tf.get_variable('Wx31', shape=(t,t),initializer=tf.contrib.layers.xavier_initializer()),
    'wd2x': tf.get_variable('Wx4', shape=(t,feature_dim), initializer=tf.contrib.layers.xavier_initializer()),
    'wd3x': tf.get_variable('Wx5', shape=(256,256), initializer=tf.contrib.layers.xavier_initializer()), 
    'outx': tf.get_variable('Wx6', shape=(t,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biasesx = {
    'bd1x': tf.get_variable('Bx3', shape=(t), initializer=tf.contrib.layers.xavier_initializer()),
    'bd11x': tf.get_variable('Bx31', shape=(t), initializer=tf.contrib.layers.xavier_initializer()),
    'bd2x': tf.get_variable('Bx4', shape=(feature_dim), initializer=tf.contrib.layers.xavier_initializer()),
    'bd3x': tf.get_variable('Bx5', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'outx': tf.get_variable('Bx6', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()), #8
}

weightsy = {
    'wd1y': tf.get_variable('Wy3', shape=(feature_dim,t), initializer=tf.contrib.layers.xavier_initializer()),
    'wd11y': tf.get_variable('Wy31', shape=(t,t),  initializer=tf.contrib.layers.xavier_initializer()),
    'wd2y': tf.get_variable('Wy4', shape=(t,feature_dim), initializer=tf.contrib.layers.xavier_initializer()),
    'wd3y': tf.get_variable('Wy5', shape=(256,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'outy': tf.get_variable('Wy6', shape=(t,n_classes),  initializer=tf.contrib.layers.xavier_initializer()), 
}
biasesy = {
    'bd1y': tf.get_variable('By3', shape=(t), initializer=tf.contrib.layers.xavier_initializer()),
    'bd11y': tf.get_variable('By31', shape=(t), initializer=tf.contrib.layers.xavier_initializer()),
    'bd2y': tf.get_variable('By4', shape=(feature_dim), initializer=tf.contrib.layers.xavier_initializer()),
    'bd3y': tf.get_variable('By5', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'outy': tf.get_variable('By6', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()), #8
}

weightsz = {
    'wd1z': tf.get_variable('Wz3', shape=(300,t),  initializer=tf.contrib.layers.xavier_initializer()),
    'wd11z': tf.get_variable('Wz31', shape=(t,t),  initializer=tf.contrib.layers.xavier_initializer()),
    'wd2z': tf.get_variable('Wz4', shape=(t,feature_dim), initializer=tf.contrib.layers.xavier_initializer()),
    'wd3z': tf.get_variable('Wz5', shape=(256,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'outz': tf.get_variable('Wz6', shape=(t,n_classes),  initializer=tf.contrib.layers.xavier_initializer()), 
}
biasesz = {
    'bd1z': tf.get_variable('Bz3', shape=(t), initializer=tf.contrib.layers.xavier_initializer()),
    'bd11z': tf.get_variable('Bz31', shape=(t), initializer=tf.contrib.layers.xavier_initializer()),
    'bd2z': tf.get_variable('Bz4', shape=(feature_dim), initializer=tf.contrib.layers.xavier_initializer()),
    'bd3z': tf.get_variable('Bz5', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),

    'outz': tf.get_variable('Bz6', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()), #8
}


def fc_netx(x, weightsx, biasesx):  


    # Fully connected layer
    #fc1 = tf.reshape(x, [-1, weightsx['wd1x'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(x, weightsx['wd1x']), biasesx['bd1x'])
    #fc1 = lrelu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weightsx['outx']), biasesx['outx'])

    fc2 = tf.reshape(fc1, [-1, weightsx['wd2x'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weightsx['wd2x']), biasesx['bd2x'])
    #fc2 = tf.nn.relu(fc2)
    #fc2 = tf.nn.dropout(fc2, keep_prob)
    return out, fc1, fc2

def fc_nety(y, weightsy, biasesy):  


    # Fully connected layer
    #fc1 = tf.reshape(y, [-1, weightsy['wd1y'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(y, weightsy['wd1y']), biasesy['bd1y'])
    #fc1 = lrelu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weightsy['outy']), biasesy['outy'])

    fc2 = tf.reshape(fc1, [-1, weightsy['wd2y'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weightsy['wd2y']), biasesy['bd2y'])
    fc2 = tf.nn.relu(fc2)
    #fc2 = tf.nn.dropout(fc2, keep_prob)
    return out,fc1, fc2

def fc_netz(z, weightsz, biasesz):  


    # Fully connected layer
    #fc1 = tf.reshape(y, [-1, weightsy['wd1y'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(z, weightsz['wd1z']), biasesz['bd1z'])
    #fc1 = lrelu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weightsz['outz']), biasesz['outz'])

    fc2 = tf.reshape(fc1, [-1, weightsz['wd2z'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weightsz['wd2z']), biasesz['bd2z'])
    fc2 = tf.nn.relu(fc2)
    #fc2 = tf.nn.dropout(fc2, keep_prob)
    return out,fc1, fc2



 
predx,fc_featsx,dux = fc_netx(x, weightsx, biasesx)
predy1,fc_featsy1,duy1 = fc_nety(y1, weightsy, biasesy)
predy2,fc_featsy2,duy2 = fc_nety(y2, weightsy, biasesy)
#predz,fc_featsz,duz = fc_netz(w, weightsz, biasesz)


px = np.zeros((n_classes,t))
py = np.zeros((t_classes,t))


W1 = tf.Variable(tf.truncated_normal([1], stddev=0.000001), name="W1")
pox = (tf.reduce_min(tf.square(tf.add(tf.subtract(fc_featsx,[w[1,:]]),0.000001)),axis = 1))
poy1 = (tf.reduce_min(tf.square(tf.add(tf.subtract(fc_featsy1,[w[1,:]]),0.000001)),axis = 1))
poy2 = (tf.reduce_min(tf.square(tf.add(tf.subtract(fc_featsy2,[w[1,:]]),0.000001)),axis = 1))

for i in range(n_classes-1):
	px1 = (tf.reduce_min(tf.square(tf.add(tf.subtract(fc_featsx,w[i+1,:]),0.000001)),axis = 1))
	px1 = tf.reshape(px1,[128,1])
	pox = tf.reshape(pox,[128,-1])
	pox = tf.concat([pox,px1],1)
	py1 = (tf.reduce_min(tf.square(tf.add(tf.subtract(fc_featsy1,w[i+1,:]),0.000001)),axis = 1))
	py1 = tf.reshape(py1,[128,1])
	poy1 = tf.reshape(poy1,[128,-1])
	poy1 = tf.concat([poy1,py1],1)
	py2 = (tf.reduce_min(tf.square(tf.add(tf.subtract(fc_featsy2,w[i+1,:]),0.000001)),axis = 1))

for i in range(t_classes-1):
	py2 = tf.reshape(py2,[128,1])
	poy2 = tf.reshape(poy2,[128,-1])
	poy2 = tf.concat([poy2,py2],1)

	



pox = tf.cast(pox,'float32')
poy1 = tf.cast(poy1,'float32')
poy2 = tf.cast(poy2,'float32')



distance1  = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(fc_featsx,fc_featsy1),2),1,keep_dims=True)))
distance2  = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(fc_featsx,fc_featsy2),2),1,keep_dims=True)))
loss0 = tf.maximum(distance1 - distance2 + alpha, 0)


#loss1 = tf.reduce_mean(tf.norm(fc_featsx))
#loss2 = tf.reduce_mean(tf.norm(fc_featsy1)) + tf.reduce_mean(tf.norm(fc_featsy2))
loss3a = tf.reduce_mean(tf.norm(tf.subtract(fc_featsx,fc_featsy1)))
#loss3b = tf.reduce_mean(tf.norm(tf.subtract(fc_featsx,fc_featsy2)))
loss3 = loss3a 
loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predx,labels=lx))
loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy1, labels=ly1))
#loss6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy2, labels=ly2))
#loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pox,labels=lx))
#loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=poy1, labels=ly1))
#loss6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=poy2, labels=ly2))
loss6 =  tf.norm(tf.subtract(tf.norm(weightsx['wd1x']),1))
loss7 =  tf.norm(tf.subtract(tf.norm(weightsy['wd1y']),1))
#loss8 = tf.norm(tf.subtract(tf.norm(weightsz['wd1z']),1))
loss9 = tf.reduce_mean(tf.norm(tf.subtract(x,duy1))) + tf.reduce_mean(tf.norm(tf.subtract(y1,dux)))
unified = fc_featsx


#cost =  loss1 + loss2 + loss3 + loss4 + loss5  + loss6 + loss7 + loss8#   # MODEL 3
cost =   0.1*loss0 + 0*loss3 + 0.1*loss4 + 1*loss5+ loss6 + loss7 #+loss8 #  + 0*loss9     # MODEL 2


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_prediction1 = tf.equal(tf.argmax(predx, 1), tf.argmax(lx, 1))
correct_prediction2 = tf.equal(tf.argmax(predy1, 1), tf.argmax(ly1, 1))
correct_prediction3 = tf.equal(tf.argmax(predy2, 1), tf.argmax(ly2, 1))

#calculate accuracy across all the given images and average them out. 
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
accuracy = (accuracy1 + accuracy2 + accuracy3)/2

init = tf.global_variables_initializer() 

print('here 3')
with tf.Session() as sess:
    sess.run(init) 
    #print(pred.get_shape())
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    final_accuracy = 0
    path = 'srcV'
    saver = tf.train.Saver()
    model_name = 'unified'
    snapshot_path = path+'/snapshots/%s/' % (model_name)
    snapshot_path_latest2 = path+'/snapshots/%s/' % (model_name)
    latest2 = tf.train.latest_checkpoint(snapshot_path_latest2)
    
    #saver.restore(sess, latest2)
    #cur_i = int(latest2[len(snapshot_path_latest2 + 'model-'):])
    #print('Restoring last models default checkpoint at %d' % cur_i)
    summary_writer = tf.summary.FileWriter('./srcV/Output', sess.graph)
    print('here 4')
    for i in range(training_iters):
        for batch in range((60384)//batch_size): #75481  60384 15097
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y1 = train_Y1[batch*batch_size:min((batch+1)*batch_size,len(train_Y1))] 
            batch_y2 = train_Y2[batch*batch_size:min((batch+1)*batch_size,len(train_Y2))] 
            batch_lx = train_Lx[batch*batch_size:min((batch+1)*batch_size,len(train_Lx))]
            batch_ly1 = train_Ly1[batch*batch_size:min((batch+1)*batch_size,len(train_Ly1))]
            batch_ly2 = train_Ly2[batch*batch_size:min((batch+1)*batch_size,len(train_Ly2))]
            batch_w = train_W[batch*batch_size:min((batch+1)*batch_size,len(train_W))] 
            # Run optimization op (backprop).

            opt = sess.run(optimizer, feed_dict={x: batch_x, y1: batch_y1, y2: batch_y2, lx: batch_lx,ly1: batch_ly1,ly2: batch_ly2, w: batch_w, keep_prob : 0.5})
            loss00, loss33,loss44, loss55, loss66, loss77, cost1, acc = sess.run([loss0, loss3,loss4, loss5, loss6, loss7, cost, accuracy], feed_dict={x:batch_x,y1: batch_y1, y2: batch_y2, lx: batch_lx,ly1: batch_ly1,ly2: batch_ly2, w: batch_w, keep_prob : 1})

            #print("Iter " + str(batch) +  ", Loss0= " + "{:.2f}".format(float(loss00)) + ", Loss3= " + "{:.2f}".format(loss33)+ ", Loss5= " + "{:.2f}".format(loss)+ ", Training Accuracy= " + "{:.2f}".format(acc))
        print("Iter " + str(i) + ", total= " + "{:.3f}".format(cost1) + ", l0= " + "{:.3f}".format(loss00)+ ", l4= " + "{:.3f}".format(loss44) + ", l5= " + "{:.3f}".format(loss55)+ ",  l6= " + "{:.3f}".format(loss66)+ ",  l7= " + "{:.3f}".format(loss77))

        #print("Optimization Finished!")
        if i%19 == 100:
        	save_model(sess, saver, i)
        '''
        # Calculate accuracy for all 10000 mnist test images
	for batch in range(len(test_X)//batch_size):
	    batch_x = test_X[batch*batch_size:min((batch+1)*batch_size,len(test_X))]
            batch_y1 = test_Y1[batch*batch_size:min((batch+1)*batch_size,len(test_Y1))] 
	    batch_y2 = test_Y2[batch*batch_size:min((batch+1)*batch_size,len(test_Y2))] 
            batch_lx = test_Ly[batch*batch_size:min((batch+1)*batch_size,len(test_Ly))]
	    batch_ly = test_Ly[batch*batch_size:min((batch+1)*batch_size,len(test_Ly))]
	    batch_w = test_W[batch*batch_size:min((batch+1)*batch_size,len(test_W))] 
	    print test_X.shape
	    print test_Y1.shape
	    print test_Ly.shape
	    print test_W.shape
            valid_loss = sess.run([cost2], feed_dict={x: batch_x,y1: batch_y1,y2: batch_y2,lt : batch_lx,w: batch_w, keep_prob : 1}) #
            #final_accuracy = final_accuracy + test_acc
            #print test_acc, batch

        #print (len(test_X)//batch_size)
	#final_accuracy = final_accuracy/(len(test_X)//batch_size)

        train_loss.append(loss)

        #test_loss.append(valid_loss)
        #train_accuracy.append(acc)
        #test_accuracy.append(test_acc)
        print("Testing loss:","{:.5f}".format(loss))
        '''
    summary_writer.close()

fc_features = None
print('Model "%s" already trained!'% model_name)
with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                #sess.run(tf.local_variables_initializer(), variable_initialization)
                print('Starting threads')
                saver = tf.train.Saver()  # Gets all variables in `graph`.
                i = load_model(sess, saver)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                num = 75481
                fc_featuresx = np.zeros((12500,t)) # to store fully-connected layer's features
                fc_featuresy = np.zeros((num,t))
                w2v = np.zeros((125,t))
                #train_idx = train_indices # to store train index
                for ind in range(0,int(X.shape[0]/batch_size)): #processing the datapoints batchwise
                    ind_s = ind*batch_size
                    ind_e = (ind+1)*batch_size
                    batch_x = Xorg[ind_s:ind_e,:]
                    batch_y1 = Yorg[ind_s:ind_e,:]
                    #batch_y2 = Y2[ind_s:ind_e,:]
                    batch_lx = Ld[ind_s:ind_e,:]
                    batch_w = W[ind_s:ind_e,:] 
                    cfeatsx, cfeatsy = sess.run([fc_featsx, fc_featsy1], feed_dict={x: batch_x, y1:batch_y1,y2:batch_y2, l1: batch_lx, w: batch_w, keep_prob : 1})
                    #cfeatsy = sess.run(, feed_dict={x: batch_x, y:batch_y, l1: batch_l, w: batch_w, keep_prob : 1})
		    #cfeatsz = sess.run(fc_featsz, feed_dict={x: batch_x, y:batch_y, l1: batch_l, w: batch_w, keep_prob : 1})
                    #print size(cpred)
                    #print cfeatsz.shape
                    fc_featuresx[ind_s:ind_e,:] = cfeatsx
                    fc_featuresy[ind_s:ind_e,:] = cfeatsy
		    #w2v[ind_s:ind_e,:] = cfeatsz
                    #print('Processing %d batch' % ind)
                # storing the extracted features in mat file
                sio.savemat('unified_features.mat', {'featuresx':fc_featuresx, 'featuresy':fc_featuresy}) #saving 'featuresz':w2v


