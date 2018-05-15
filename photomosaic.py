#!/usr/bin/python3
__author__ = "Fredrik Opeide"

import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys
import getopt


_tile_histograms = []
_tile_flat = []


def create_mosaic(path_tiles, path_target, img_size, tile_size):
    tiles = load_downsampled(path_tiles, tile_size)

    target = load_image(path_target, img_size)

    #make divisible by tile size
    result_shape = np.array(target.shape) / np.array(tile_size+[1])
    result_shape = result_shape.astype(dtype=int)*(np.array(tile_size+[1]))
    result_img = np.zeros(shape=result_shape, dtype=np.uint8)

    target_chunkinator = generate_image_chunks(target, tile_size)

    i = 0   #used for estimating progress
    t = time.time()
    print('matching tiles')
    for img_chunk, chunk_location in target_chunkinator:
        i += 1
        best_tile = get_best_match(img_chunk, tiles)
        result_img = add_tile_to_image(result_img, best_tile, chunk_location)
        if time.time() > t + 1:
            t = time.time()
            print('\t{}%'.format(int(100*i*np.product(tile_size)/np.product(result_img.shape[:2]))))
    print('\t100%')
    return result_img


def generate_image_chunks(img, chunk_size):
    for loc0 in range(0,img.shape[0]-chunk_size[0], chunk_size[0]):
        for loc1 in range(0, img.shape[1]-chunk_size[1], chunk_size[1]):
            chunk_location = [loc0, loc1]
            img_chunk = img[loc0:loc0+chunk_size[0], loc1:loc1+chunk_size[1], :]
            yield img_chunk, chunk_location


def add_tile_to_image(img, tile, tile_location):
    loc0 = tile_location[0]
    loc1 = tile_location[1]
    off0 = tile.shape[0]
    off1 = tile.shape[1]
    img[loc0:loc0+off0, loc1:loc1+off1, :] = tile
    return img



#todo: Supress similar matces within a radius
def get_best_match(target, tiles, loss_type='norm'):
    if loss_type == 'norm':
        target_flat = target.flatten()/255.0
        if not _tile_flat:    #simple dynamic programming
            for tile in tiles:
                tile_flat = np.array(tile).flatten() / 255.0
                _tile_flat.append(tile_flat)


        lowest_diff = None
        best_flat = None
        for tile_flat in _tile_flat:
            diff = np.linalg.norm(target_flat-tile_flat)
            if (lowest_diff is None) or diff < lowest_diff:
                lowest_diff = diff
                best_flat = tile_flat
        best_tile = (best_flat*255).astype(np.uint8)
        best_tile = np.reshape(best_tile, target.shape)
        best_tile = np.array(Image.blend(Image.fromarray(best_tile), Image.fromarray(target), alpha=0.25))
        return best_tile

    elif loss_type == 'histogram':
        if not _tile_histograms:    #simple dynamic programming
            for img in tiles:
                hist, _ = np.histogram(img.flatten(), bins=32, range=[0, 255])
                _tile_histograms.append([img, hist])
        hist_target, _ = np.histogram(target.flatten(), bins=32, range=[0, 255])

        best_similarity = -1
        closest_img = None
        for img, hist_img in _tile_histograms:
            similarity = histogram_intersection(hist_target, hist_img)
            if similarity > best_similarity:
                best_similarity = similarity
                closest_img = img
        return closest_img


def histogram_intersection(h1, h2):
    minima = np.minimum(h1, h2)
    intersection = np.true_divide(np.sum(minima), np.sum(h2))
    return intersection


def load_downsampled(source, size):
    print('Loading images and downsampling,', source)
    images = []
    for source_img_path in glob.glob(source+'*.jpg'):
        source_img = load_image(source_img_path)

        factor0 = int(source_img.shape[0] / size[0])
        factor1 = int(source_img.shape[1] / size[1])
        source_img_downsampled = source_img[::factor0, ::factor1, :]
        source_img_downsampled = source_img_downsampled[:size[0], :size[1], :]

        images.append(source_img_downsampled)
    return images


def load_image(path, img_size=None):
    img = Image.open(path)
    if img_size is not None:
        img = img.resize(img_size, Image.ANTIALIAS)
    return np.array(img)


def show_image(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    long_args = ['source=', 'target=', 'save_path=', 'img_size_x=', 'img_size_y=', 'tile_size=']
    opts, _ = getopt.getopt(sys.argv[1:], '', long_args)

    missing_args = ['--source', '--target', '--save_path', '--tile_size', '--img_size_x', '--img_size_y']
    received_args = {}
    for arg, val in opts:
        try:
            received_args[arg] = val
            missing_args.remove(arg)
        except:
            raise Exception('Unexpected argument {}, expecting: {}'.format(arg, long_args))
    if missing_args:
        raise Exception('Missing argument: {}'.format(missing_args))

    print(received_args)
    path_tiles = received_args['--source']
    path_target = received_args['--target']
    path_result = received_args['--save_path']
    img_size = (int(received_args['--img_size_x']), int(received_args['--img_size_y']))
    n = int(received_args['--tile_size'])
    tile_size = [n, n]

    mosaic = create_mosaic(path_tiles, path_target, img_size, tile_size)
    show_image(mosaic)
    mosaic_pil = Image.fromarray(mosaic)
    mosaic_pil.save(path_result)

