from keras.models import load_model
from keras.models import Model

# Load the saved model
model = load_model('C:/Users/nvraj/Downloads/S8 Project/best_weights.hdf5')

# Make modifications to the model
# For example, modify the name of the problematic layer
for layer in model.layers:
    if layer.name == 'conv1/conv':
        layer.name = 'conv1_conv_modified'  # Modify the name

# Rebuild the model to update internal references
# Create a new model with the same architecture and modified layer names
input_layer = model.input
output_layer = model.output
modified_model = Model(inputs=input_layer, outputs=output_layer)

# Save the modified model to a new file
modified_model.save('C:/Users/nvraj/Downloads/S8 Project/best_weights_modified.hdf5')
