import tensorflow as tf
class DS1(tf.keras.layers.Layer):
    def __init__(self, units, input_dim):
        super(DS1, self).__init__()
        self.w=self.add_weight(
            name='Prototypes',
            shape=(units, input_dim),
            initializer='random_normal',
            trainable=True
        )
        
        self.units=units
                    
    def call(self, inputs):
        for i in range(self.units):
          if i==0:
            un_mass_i=tf.subtract(self.w[i,:], inputs, name=None)
            un_mass_i=tf.square(un_mass_i, name=None)
            un_mass_i=tf.reduce_sum(un_mass_i, -1, keepdims=True)
            un_mass = un_mass_i

          if i>=1:
            un_mass_i=tf.subtract(self.w[i,:], inputs, name=None)
            un_mass_i=tf.square(un_mass_i, name=None)
            un_mass_i=tf.reduce_sum(un_mass_i, -1, keepdims=True)
            un_mass=tf.concat([un_mass, un_mass_i], -1)
        return un_mass

class DS1_activate(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(DS1_activate, self).__init__()
        self.xi=self.add_weight(
            name='xi',
            shape=(1, input_dim),
            initializer='random_normal',
            trainable=True
        )
        
        self.eta=self.add_weight(
            name='eta',
            shape=(1, input_dim),
            initializer='random_normal',
            trainable=True
        )
        
        self.input_dim=input_dim
                    
    def call(self, inputs):
        gamma=tf.square(self.eta, name=None)
        alpha=tf.negative(self.xi, name=None)
        alpha=tf.exp(alpha, name=None)+1
        alpha=tf.divide(1, alpha, name=None)
        si=tf.multiply(gamma, inputs, name=None)
        si=tf.negative(si, name=None)
        si=tf.exp(si, name=None)
        si=tf.multiply(si, alpha, name=None)
        si = si / (tf.reduce_max(si, axis = -1, keepdims=True) + 0.0001)
        return si

class DS2(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class):
        super(DS2, self).__init__()
        self.beta=self.add_weight(
            name='beta',
            shape=(input_dim, num_class),
            initializer='random_normal',
            trainable=True
        )
        
        self.input_dim=input_dim
        self.num_class=num_class
                    
    def call(self, inputs):
        beta=tf.square(self.beta, name=None)
        beta_sum=tf.reduce_sum(beta, -1, keepdims=True)
        u=tf.divide(beta, beta_sum, name=None)
        inputs_new=tf.expand_dims(inputs, -1)
        for i in range(self.input_dim):
          if i==0:
            mass_prototype_i=tf.multiply(u[i,:], inputs_new[:,i], name=None)
            mass_prototype=tf.expand_dims(mass_prototype_i, -2)
          if i>0:
            mass_prototype_i=tf.expand_dims(tf.multiply(u[i,:], inputs_new[:,i], name=None), -2)
            mass_prototype=tf.concat([mass_prototype, mass_prototype_i], -2)
        mass_prototype=tf.convert_to_tensor(mass_prototype)
        return mass_prototype

class DS2_omega(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class):
        super(DS2_omega, self).__init__()
        self.input_dim=input_dim
        self.num_class=num_class
                    
    def call(self, inputs):
        mass_omega_sum=tf.reduce_sum(inputs, -1, keepdims=True)
        mass_omega_sum=tf.subtract(1., mass_omega_sum[:,:,0], name=None)
        mass_omega_sum=tf.expand_dims(mass_omega_sum, -1)
        mass_with_omega=tf.concat([inputs, mass_omega_sum], -1)
        return mass_with_omega
    
class DS3_Dempster(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class):
        super(DS3_Dempster, self).__init__()
        self.input_dim=input_dim
        self.num_class=num_class
        
    def call(self, inputs):
        m1=inputs[:,0,:]
        omega1=tf.expand_dims(inputs[:,0,-1],-1)
        for i in range (self.input_dim-1):
            m2=inputs[:,(i+1),:]
            omega2=tf.expand_dims(inputs[:,(i+1),-1], -1)
            combine1=tf.multiply(m1, m2, name=None)
            combine2=tf.multiply(m1, omega2, name=None)
            combine3=tf.multiply(omega1, m2, name=None)
            combine1_2=tf.add(combine1, combine2, name=None)
            combine2_3=tf.add(combine1_2, combine3, name=None)
            combine2_3 = combine2_3 / tf.reduce_sum(combine2_3, axis = -1, keepdims=True)#后加的
            m1=combine2_3
            omega1=tf.expand_dims(combine2_3[:,-1], -1)
            #m1=combine2_3[:,:-1]
            #omega1=omega1*omega2
            #m1=tf.concat([m1, omega1], -1)
        return m1

class DS3_normalize(tf.keras.layers.Layer):
    def __init__(self):
        super(DS3_normalize, self).__init__()
        
    def call(self, inputs):
        mass_combine_normalize = inputs / tf.reduce_sum(inputs, axis = -1, keepdims=True)
        return mass_combine_normalize
