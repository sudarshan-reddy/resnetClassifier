from load_celeb_a import load_celeb_data
import keras
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras import backend as K

if __name__ == '__main__' : 
    sess = tf.Session()
    K.set_session(sess)

    train_gen, val_gen = load_celeb_data(224, 224)
    model = keras.applications.resnet50.ResNet50()
    classes = train_gen.class_indices
    
    model.layers.pop()

    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[:30]:
        layer.trainable = True
    last = model.layers[-1].output

    x = Dense(len(classes), activation="softmax", name="inferenceLayer")(last)

    finetuned_model = Model(model.input, x)

    finetuned_model.compile(optimizer=Adam(lr=1e-04), loss='categorical_crossentropy', metrics=['accuracy'])

    for c in list(train_gen.class_indices):
        classes[train_gen.class_indices[c]] = c
    finetuned_model.classes = classes

    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)

    early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)

    finetuned_model.fit_generator(train_gen, 
                                  steps_per_epoch=100, epochs=1,
                                  callbacks=[early_stopping, checkpointer],
                                  validation_data=val_gen,
                                  validation_steps=100
                                 )
    finetuned_model.save('resnet50_final.h5')

    builder = tf.saved_model.builder.SavedModelBuilder("forGo")
    builder.add_meta_graph_and_variables(sess, ["tags"])
    builder.save()
    sess.close()
