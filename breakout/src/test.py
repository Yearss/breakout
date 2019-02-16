

import tensorflow as tf


a = tf.placeholder(name='input', dtype=tf.float32, shape=[])

b = tf.summary.scalar(tensor=a, name="test")

sum_ops = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter("../tmp1")

with tf.Session() as sess:

    for x in range(10):
        sum_log = sess.run(sum_ops, feed_dict={
            a: x
        })
        summary_writer.add_summary(sum_log)




