from keras.preprocessing.image import ImageDataGenerator

train_directory = "/home/sudarshan/dev/deeplearning/cnn_finetune/CelebA/classes/train"
valid_directory ="/home/sudarshan/dev/deeplearning/cnn_finetune/CelebA/classes/valid"
train_batch_size = "50"
valid_batch_size = "50"

def load_celeb_data(img_rows=224, img_cols=224):
    train_batches = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    ).flow_from_directory(train_directory,target_size=(img_rows,img_cols),
                          shuffle=True,
                          batch_size=train_batch_size, classes=('male',
                                                                'female'))
    valid_batches = ImageDataGenerator().flow_from_directory(valid_directory,target_size=(img_rows,img_cols),
                          shuffle=True,
                          batch_size=valid_batch_size,classes=('male','female'))

    return train_batches, valid_batches

