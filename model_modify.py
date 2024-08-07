from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# load the original model
model = load_model('HiLo220511_fromYH.h5')

# print the model summary
# model.summary()

# Obtain the input layer of the original model
inputs = model.input

# Obtain the output layer of the original model
x = model.layers[-2].output

# Setup the new output layer
new_output = Dense(units=20, activation='relu')(x)

# Setup the new model
new_model = Model(inputs=inputs, outputs=new_output)

# Freeze the layers of the original model
for layer in new_model.layers[:-1]:
    layer.trainable = False

# Compile the new model
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the summary of the new model
new_model.summary()

# Save the new model
new_model.save('HiLo220511_fromYH_v2.h5')
