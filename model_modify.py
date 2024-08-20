from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# load the original model
model = load_model('HiLo220511_fromYH.h5')

# print the model summary
# model.summary()

# Obtain the input layer of the original model
inputs = model.input

# Obtain the output layer of the original model
x = model.layers[-2].output

# Setup the new output layer
# ReLu is easy to generate high intensity value
# But it's easy to obtain the image of over-saturated.
# Using "sigmoid" or "tanh" is better to control the intensity range.
# Using L2 regularization can also control the weights of the model.
new_output = Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.00001))(x)
# new_output = Dense(units=1, activation='sigmoid')(x)

# Setup the new model
new_model = Model(inputs=inputs, outputs=new_output)

# Freeze all layers except the last one
for layer in new_model.layers[:-1]:
    layer.trainable = False

# Compile the new model
# Adjust the learning rate can also avoid the over-saturated problem
new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mean_absolute_error')
# new_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_absolute_error')


# Print the summary of the new model
# new_model.summary()

# Save the new model
new_model.save('HiLo220511_fromYH_v2.h5')
