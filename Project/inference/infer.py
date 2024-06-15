import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from keras.utils import img_to_array
import matplotlib.pyplot as plt
import os

def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()

def infer(model, image_path, save_path):
    original_image = Image.open(image_path)
    image = img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output = model.predict(image, verbose=0)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    original_image = Image.fromarray(np.uint8(original_image))
    enhanced_image = Image.fromarray(np.uint8(output_image))

    if save_path:
        enhanced_image.save(save_path)
    
    return original_image, enhanced_image

def infer_and_save(model, test_low_light_images, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for low_light_image in test_low_light_images:
        file_name = os.path.basename(low_light_image)
        save_path = os.path.join(save_dir, file_name)
        original_image, enhanced_image = infer(model, low_light_image, save_path)
        plot_results(
            [original_image, enhanced_image],
            ["Original", "MIRNet Enhanced"],
            (20, 12),
        )
