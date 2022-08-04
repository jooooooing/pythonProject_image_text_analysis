import tensorflow as tf



#---------------------------------------------------------------------------------------------------------- 입력
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



#---------------------------------------------------------------------------------------------------------- 모델
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_model = tf.matmul(x, W) + b



#---------------------------------------------------------------------------------------------------------- 라벨
y = tf.placeholder(tf.float32, [None, 10])



#---------------------------------------------------------------------------------------------------------- loss와 optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_model))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # learning_rate = 0.01



#---------------------------------------------------------------------------------------------------------- 세션
sess = tf.Session()


sess.run(tf.global_variables_initializer())


for epoch in range(25):
    avg_cost = 0.

    total_batch = int(mnist.train.num_examples / 100)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += c / total_batch

    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))


print("최적화 완료")


correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


#---------------------------------------------------------------------------------------------------- 모델 파라미터 저장
model_path = "/tmp/model.saved" # 모델 파라미터를 저장할 경로와 파일 이름
saver = tf.train.Saver()

save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)

sess.close()