from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import skimage
import cv2
import os


base_config = {
    'root_directory': './tmp/MICRST',
    'width': 28,
    'height': 28,
    'chars': [
        '1', '2', '3', '4',
        '5', '6', '7', '8',
        '9', '0', 'A', 'B',
        'C', 'D'
    ],
    'textures_directory': 'tmp/textures',
    'paper_sigma_span': [14, 22],
    'background_distribution': {
        'paper': 70,
        'texture': 15,
        'cyclic': 10
    },
    'font_size_span': [25, 32],
    'blur_probability': 0.22,
    'noise_probability': 0.32,
    'random_affine_probability': 0.233
}


def make_dataset(config, number_of_samples=10000, force=False):
    # Prepare the directories to store images and labels
    root_dir = config['root_directory']

    if os.path.exists(root_dir) and not force:
        print('Dataset exists, add `force=True` to rebuild!')
        return

    images_dir = prepare_directory(root_dir)

    # Prepare output container
    records = []

    for it in tqdm(range(number_of_samples)):
        # Synthetic generation
        img, ground_truth = make_micrst_sample(config)

        # Make the numbering easily sortable
        number = 1_000_000 + it
        savepath = os.path.join(images_dir, f'MICRST_{number}.png')

        cv2.imwrite(savepath, img)
        record = {'path': savepath, 'label': ground_truth}
        records.append(record)

    df = pd.DataFrame(records)
    labels_path = os.path.join(root_dir, 'labels.csv')
    df.to_csv(labels_path, index=False)


def make_preview(config):
    columns = []
    for it in range(10):
        column = []
        for jt in range(6):
            # Don't care about the printed symbol
            micrst_sample, _ = make_micrst_sample(config)

            # Add border for esthetic reasons
            micrst_sample = make_border(micrst_sample)

            column.append(micrst_sample)
        column = np.vstack(column)
        columns.append(column)

    display_grid = np.hstack(columns)
    return display_grid


def prepare_directory(root_dir):
    images_dir = os.path.join(root_dir, 'image')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        os.makedirs(images_dir)
    else:
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        else:
            query = os.path.join(images_dir, '*')
            old_files = glob(query)
            for f in old_files:
                os.remove(f)

    return images_dir


def make_micrst_sample(config):
    img = make_background(config)
    img, ground_truth = draw_character(img, config)

    img = introduce_obfuscations(img, config)

    return img, ground_truth


def introduce_obfuscations(img, config):
    img = add_noise(img, config)

    # If anything, I think it's better to blur after nois
    img = random_blur(img, config)

    img = random_affine(img, config)

    return img


def random_affine(img, config):
    probability = config['random_affine_probability']
    if np.random.random() > probability:
        return img

    rows, cols, _ = img.shape

    # Original corners coordinates
    starting_points = [
        [0, 0],
        [0, rows],
        [cols, 0]
    ]

    # Coordinates shifted randomly left/right and up/down
    transformed_points = []
    for p in starting_points:
        # The -4::4 span is arbitrary and could be configured
        new_point = [p[0] + np.random.randint(-4, 5), p[1] + np.random.randint(-4, 5)]
        transformed_points.append(new_point)

    pts1 = np.float32(starting_points)
    pts2 = np.float32(transformed_points)

    # Transformation matrix
    M = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))

    return img


def random_blur(img, config):
    probability = config['blur_probability']
    if np.random.random() > probability:
        # No blur
        return img

    # Kernel size could be configurable as well,
    # but for 28x28 going over 3 is practically destroying
    # the information we'd like to extract
    kernel_size = np.random.randint(1, 3)
    img = cv2.blur(img, (kernel_size, kernel_size))

    return img


def add_noise(img, config):
    probability = config['noise_probability']
    if np.random.random() < probability:
        return img

    # More randomness
    noise_modes = [
        "gaussian", "localvar", "poisson",
        "salt", "pepper", "s&p", "speckle"
    ]
    noise_mode = np.random.choice(noise_modes)

    # Go back and forth from np/cv2 to skimage formats ...
    img = img / 255.
    img = 255 * skimage.util.random_noise(img, mode=noise_mode)

    img = img.astype(np.uint8)

    return img


def draw_character(img, config):
    # Prepare path
    font_path = "./micr-encoding.regular.ttf"
    font_size = np.random.randint(*config['font_size_span'])
    font = ImageFont.truetype(font_path, font_size)

    # Prepare image
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # Random shift
    x0 = np.random.randint(10) + 2
    y0 = np.random.randint(10) + 2

    # MICR is printed in black only, we could add
    # a little random gray variations, feel free to experiment
    color_black = (0, 0, 0)

    # Choose one of the symbols
    character = np.random.choice(config['chars'])
    draw.text((x0, y0), character, font=font, fill=color_black)

    # Convert back to OpenCV/NumPy format
    img = np.array(img_pil, dtype=np.uint8)

    # We have to keep the information about
    # the image content to teach our algorithms
    return img, character


def choice_from_distribution(distribution):
    # Results space
    possibilities = list(distribution.keys())

    # Get normalized probabilities
    probabilities = list(distribution.values())
    probabilities = [p/sum(probabilities) for p in probabilities]

    result = np.random.choice(possibilities, p=probabilities)

    return result


def preview_backgrounds(config):
    backgrounds = []
    for it in range(4):
        layer = []
        for jt in range(4):
            background = make_border(make_background(config))
            layer.append(background)
        layer = np.vstack(layer)
        backgrounds.append(layer)
    backgrounds = np.hstack(backgrounds)

    plt.imshow(backgrounds)
    plt.show()


def make_background(config):
    background_type = choice_from_distribution(config['background_distribution'])
    if background_type == 'paper':
        img = make_paper_background(config)
    if background_type == 'texture':
        img = make_texture_background(config)
    if background_type == 'cyclic':
        img = make_cyclic_background(config)

    return img


def make_border(img, width_px=1):
    img[:, -width_px:, :] = 0
    img[:, :width_px, :] = 0
    img[-width_px:, :, :] = 0
    img[:width_px, :, :] = 0
    return img


def make_texture_background(config):
    query = os.path.join(config['textures_directory'], '*g')
    texture_paths = glob(query)

    # Select one
    texture_path = np.random.choice(texture_paths)
    texture = cv2.imread(texture_path)

    # Cut out a smaller fragment in size of final image
    img = random_crop(texture, config['width'], config['height'])

    return img


def random_crop(img, width, height):
    in_height, in_width, _ = img.shape
    h_sta = np.random.randint(0, in_height - height + 1)
    w_sta = np.random.randint(0, in_width - width + 1)

    h_end = h_sta + height
    w_end = w_sta + width

    img = img[h_sta: h_end, w_sta: w_end, ...]

    return img


def make_paper_background(config):
    """
    Consequently applies noise patterns to the original image from big to small.
    source: https://stackoverflow.com/a/51653017

    sigma: defines bounds of noise fluctuations
    turbulence: defines how quickly big patterns will be replaced with the small ones. The lower
        value - the more iterations will be performed during texture generation.
    """
    paper_w, paper_h = 1024, 1024
    image = 255 * np.ones([paper_h, paper_w], np.uint8)

    sigma = np.random.randint(*config['paper_sigma_span'])
    turbulence = 2
    result = image.astype(float)
    cols, rows = image.shape
    ratio = cols
    while not ratio == 1:
        result += paper_noise(cols, rows, ratio, sigma=sigma)
        ratio = (ratio // turbulence) or 1
    cut = np.clip(result, 0, 255).astype(np.uint8)
    cut = cv2.cvtColor(cut, cv2.COLOR_GRAY2BGR)

    out = random_crop(cut, config['width'], config['height'])
    return out


def paper_noise(width, height, ratio=1, sigma=15):
    """
    The function generates an image, filled with gaussian nose. If ratio parameter is specified,
    noise will be generated for a lesser image and then it will be upscaled to the original size.
    In that case noise will generate larger square patterns. To avoid multiple lines, the upscale
    uses interpolation.

    ratio: the size of generated noise "pixels"
    sigma: defines bounds of noise fluctuations
    """
    mean = 0
    assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(width, ratio)
    assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(height, ratio)

    h = int(height / ratio)
    w = int(width / ratio)

    result = np.random.normal(mean, sigma, (w, h, 1))
    if ratio > 1:
        result = cv2.resize(result, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    return result.reshape((width, height))


def make_cyclic_background(config):
    PI = 4
    x = np.linspace(-PI, PI, 1001)
    y = np.linspace(-PI, PI, 1001)
    X, Y = np.meshgrid(x, y)

    howmany = np.random.randint(3, 12)
    frequencies = 38 + 29 * np.random.random(howmany)
    out = np.zeros_like(X)
    for f in frequencies:
        if np.random.random() > 0.5:
            out += np.cos(f * X)**2
        if np.random.random() > 0.5:
            out += np.sin(f * Y)**2

    out -= out.min()
    out /= out.max()
    out = 150 + 105 * out
    out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    out = random_crop(out, config['width'], config['height'])

    return out
