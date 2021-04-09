import tensorflow as tf
class DM_test(tf.keras.layers.Layer):
    def __init__(self, num_class, num_set, nu):
        super(DM_test, self).__init__()
        self.num_class = num_class
        self.nu = nu
        self.utility_matrix=self.add_weight(
            name='utility_matrix',
            shape=(num_set, num_class),
            initializer='random_normal',
            trainable=False
        )
                    
    def call(self, inputs):
        for i in range(len(self.utility_matrix)):
          if i==0:
            precise = tf.multiply(inputs[:, 0: self.num_class], self.utility_matrix[i], name=None)
            precise = tf.reduce_sum(precise, -1, keepdims=True)
            omega_1 = tf.multiply(inputs[:, -1], tf.reduce_max(self.utility_matrix[i]), name=None)
            omega_2 = tf.multiply(inputs[:, -1], tf.reduce_min(self.utility_matrix[i]), name=None)
            omega = tf.expand_dims(self.nu*omega_1+(1-self.nu)*omega_2, -1)
            omega = tf.dtypes.cast(omega, tf.float32)
            utility = tf.add(precise, omega, name=None)

          if i>=1:
            precise = tf.multiply(inputs[:, 0: self.num_class], self.utility_matrix[i], name=None)
            precise = tf.reduce_sum(precise, -1, keepdims=True)
            omega_1 = tf.multiply(inputs[:, -1], tf.reduce_max(self.utility_matrix[i]), name=None)
            omega_2 = tf.multiply(inputs[:, -1], tf.reduce_min(self.utility_matrix[i]), name=None)
            omega = tf.expand_dims(self.nu*omega_1+(1-self.nu)*omega_2, -1)
            omega = tf.dtypes.cast(omega, tf.float32)
            utility_i = tf.add(precise, omega, name=None)
            utility = tf.concat([utility, utility_i], -1)
        return utility