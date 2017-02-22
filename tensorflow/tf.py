import tensorflow as tf

sess = tf.Session()

def constant_add():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly
    print(node1, node2)
    node3 = tf.add(node1,node2)
    sess = tf.Session()
    print(sess.run(node3))
def placeholder_calculate(v1,v2):
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    c = a + b
    d = c * 3
    return sess.run(d,{a:v1,b:v2})

def variable_graph(X):
    W = tf.Variable([.3],tf.float32)
    b = tf.Variable([-.3],tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess.run(linear_model,{x:X})

def linear_model_square_loss(X,Y):
    # define a model
    W = tf.Variable([.3],tf.float32)
    b = tf.Variable([-.3],tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    linear_model = W * x + b

    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    '''
    print "loss with W=.3 and b=-.3:", sess.run(loss,{x:X,y:Y})
    fixW = tf.assign(W,[-1.])
    fixb = tf.assign(b,[1.])
    sess.run([fixW,fixb])
    print "loss after changed W,b with W=-1 and b = 1.:", sess.run(loss,{x:X,y:Y})
    '''

    print "train model with SGD"
    for i in range(1000):
        sess.run(train, {x:X, y:Y})
        print i,":",sess.run([W,b]),"loss=",sess.run(loss,{x:X,y:Y})
    print sess.run([W,b])
    print Y-sess.run(linear_model,{x:X})

    

    
print "====================LOG================"
X,Y=[1,2,3,4],[7,8,9,10]
linear_model_square_loss(X,Y)



     
