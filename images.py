import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

def display_image(image, title='', cmap=''):
    imgplot = plt.imshow(image)
    if cmap:
        imgplot.set_cmap(cmap)
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
                 ].ravel()

''' normalize a set of image slices, for use in autoencoder with sigmoid
    activation. this requires input to be between 0 and 1 because the output will
    be within that range; additionally, to ensure that derivatives are not too
    small and thus slowing down learning, tighten the range to 0.1 to 0.9. '''
def normalize_image_slices(image_slices):
    demeaned_image_slices = image_slices - np.mean(image_slices)
    stdev_limit = 3. * np.std(demeaned_image_slices)
    raw_normalized_image_slices = \
        np.minimum(np.maximum(demeaned_image_slices, -stdev_limit), stdev_limit) / stdev_limit
    return 0.4 * raw_normalized_image_slices + 0.5

def opt_rescale(image, rescale):
    if not rescale:
        return image
    else:
        max_pixel = np.max(np.abs(image))
        return image / max_pixel
    
def display_image_grid(images, \
                       image_size, \
                       num_images_per_row_col, \
                       rescale=True, \
                       rgb=False
                      ):
    num_pixels = image_size * image_size
    max_index = image_size * num_images_per_row_col
    if rgb:
        final_image = np.zeros((max_index, max_index, 3))
    else:
        final_image = np.zeros((max_index, max_index))
    xmin, ymin, xmax, ymax = 0, 0, image_size, image_size
    for image in images:
        if rgb:
            for color_index in xrange(3):
                sub_image = image[(color_index * num_pixels): ((color_index + 1) * num_pixels)]
                reshaped = opt_rescale(np.reshape(sub_image, (image_size, image_size), 'F'), rescale)
                final_image[xmin:xmax, ymin:ymax, color_index] = reshaped
        else:
            reshaped = opt_rescale(np.reshape(image, (image_size, image_size), 'F'), rescale)
            final_image[xmin:xmax, ymin:ymax] = reshaped
        if ymax == max_index:
            ymin, ymax = 0, image_size
            xmin = xmin + image_size
            xmax = xmax + image_size
        else:
            ymin = ymin + image_size
            ymax = ymax + image_size
    if rescale:
        final_image -= np.min(final_image)
        final_image /= np.max(final_image)
    display_image(final_image)
