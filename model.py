import tensorflow as tf

class Model(object):
    def add_placeholders(self):
        self.lr = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32, [None, 6])
        self.y = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

    def create_feed_dict(self, inputs_batch, is_training, labels_batch=None, lr=None):
        feed_dict = {}
        feed_dict[self.X] = inputs_batch
        feed_dict[self.is_training] = is_training
        if lr is not None:
            feed_dict[self.lr] = lr
        if labels_batch is not None:
            feed_dict[self.y] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        h1 = tf.layers.dense(self.X, 6, activation=tf.nn.relu)
        h1 = tf.layers.dense(h1, 6, activation=tf.nn.relu)
        pred = tf.layers.dense(h1, 1)
        return pred

    def add_loss_op(self, pred):
        loss = tf.reduce_mean(tf.nn.l2_loss(self.y - pred))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, rate):
        feed = self.create_feed_dict(inputs_batch, True, labels_batch=labels_batch, lr=rate)
        _, loss, pred = sess.run([self.train_op, self.loss, self.pred], feed_dict=feed)
        return loss, pred

    def predict_on_batch(self, sess, inputs_batch, labels_batch=None):
        feed = self.create_feed_dict(inputs_batch, False, labels_batch=labels_batch)
        loss, pred = sess.run([self.loss, self.pred], feed_dict=feed)
        return loss, pred

    def __init__(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
