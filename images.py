import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

def display_image(image, title=''):
    imgplot = plt.imshow(image)
    imgplot.set_cmap('binary')
    plt.title(title)
    plt.show()

''' given a set of images, select an image at random and select a random slice of it'''
def generate_random_image_slice(images, height, width):
    height_images, width_images, num_images = images.shape
    image_index = random.randint(0, num_images)
    image_height = random.randint(0, height_images - height)
    image_width = random.randint(0, width_images - width)
    return images[image_height:(image_height + height), \
                  image_width:(image_width + width), \
                  image_index \
                 ].flatten()
                 
def display_image_grid(images, image_size, num_images_per_row_col):
    max_index = image_size * num_images_per_row_col
    final_image = np.zeros((max_index, max_index))
    xmin, ymin, xmax, ymax = 0, 0, image_size, image_size
    for image in images:
        reshaped = np.reshape(image, (image_size, image_size), 'F')
        max_pixel = np.max(reshaped)
        final_image[xmin:xmax, ymin:ymax] = reshaped / max_pixel
        if ymax == max_index:
            ymin, ymax = 0, image_size
            xmin = xmin + image_size
            xmax = xmax + image_size
        else:
            ymin = ymin + image_size
            ymax = ymax + image_size
    display_image(final_image)
