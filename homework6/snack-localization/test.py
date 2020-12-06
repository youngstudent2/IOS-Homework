from keras.models import Model, load_model
classifier_model = load_model("checkpoints/multisnacks-0.7162-0.8419.hdf5")
classifier_model.summary()
base_model = classifier_model.layers[0]
print(base_model)