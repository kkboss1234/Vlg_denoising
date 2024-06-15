import tensorflow as tf
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from model.mirnet import mirnet_model
from data.dataset import get_dataset


def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))

def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


def train_model(train_low_light_images, train_enhanced_images, val_low_light_images, val_enhanced_images, image_size, batch_size, epochs, model_save_path):
    train_dataset = get_dataset(train_low_light_images, train_enhanced_images)
    val_dataset = get_dataset(val_low_light_images, val_enhanced_images)

    model = mirnet_model(num_rrg=2, num_mrb=1, channels=32)
    optimizer = optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=charbonnier_loss,
        metrics=[peak_signal_noise_ratio],
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[
            ReduceLROnPlateau(
                monitor="val_peak_signal_noise_ratio",
                factor=0.5,
                patience=5,
                verbose=1,
                min_delta=1e-7,
                mode="max",
            )
        ],
    )
    

    
    return history, model
