from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
import cv2
import os


base_config = {
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
        'paper': 30,
        'texture': 70,
        'cyclic': 10
    },
    'font_size_span': [15, 30]
}


def draw_character(config):
    img = make_background(config)

    font_path = "./micr-encoding.regular.ttf"
    font_size = np.random.randint(*config['font_size_span'])
    font = ImageFont.truetype(font_path, font_size)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    x0 = np.random.randint(10) + 2
    y0 = np.random.randint(10) + 2
    color_black = (0, 0, 0)
    text = np.random.choice(config['chars'])
    draw.text((x0, y0),  text, font=font, fill=color_black)

    digit = np.array(img_pil, dtype=np.uint8)

    return digit


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
