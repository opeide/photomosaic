#!/usr/bin/python3
__author__ = "Fredrik Opeide"

import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys
import getopt
from collections import defaultdict
import pickle

_tile_histograms = []
_tile_flat = []
_placed_tiles = defaultdict(str)
_disc_offsets = []

def create_mosaic(path_tiles, path_target, tile_size, tile_resolution, duplicate_radius):
    #searches using chunks of tile size, adds tiles with their true (higher) resolution to image mosaic
    target = load_image(path_target)
    tiles_lowres, tiles_hires, tile_paths = load_tiles(path_tiles, tile_size, tile_resolution)

    #crop to multiple of hires tile resolution
    result_num_tiles = np.floor_divide(np.array(target.shape), np.array([tile_size, tile_size, 1])) #1 is added for the rgb channel
    result_shape = np.multiply(result_num_tiles, [tile_resolution, tile_resolution, 1])
    result_img = np.zeros(shape=result_shape, dtype=np.uint8)

    target_chunkinator = generate_image_chunks(target, tile_size)

    iter = 0   #used for estimating progress
    t = time.time()
    print('matching tiles')
    for img_chunk, chunk_indices in target_chunkinator:
        iter += 1
        best_tile = get_best_tile(img_chunk, tiles_lowres, tiles_hires, tile_paths, chunk_indices, duplicate_radius)
        result_img = add_tile_to_image(result_img, best_tile, chunk_indices)
        if time.time() > t + 1:
            t = time.time()
            print('\t{}%'.format(int(100*iter*np.product([tile_resolution,tile_resolution])/np.product(result_img.shape[:2]))))
    print('\t100%')
    return result_img


def generate_image_chunks(img, chunk_size):
    for m in range(img.shape[0]//chunk_size):
        for n in range(img.shape[1]//chunk_size):
            img_chunk = img[m*chunk_size:(m+1)*chunk_size, n*chunk_size:(n+1)*chunk_size, :]
            yield img_chunk, (m, n)


def add_tile_to_image(img, tile, chunk_indices):
    resolution = tile.shape[0]
    m, n = chunk_indices
    img[m*resolution:(m+1)*resolution, n*resolution:(n+1)*resolution, :] = tile
    return img

def tile_within_radius(tile_path, indices):
    m0, n0 = indices
    for dm, dn in _disc_offsets:
            key = (m0+dm, n0+dn)
            if _placed_tiles[key] == tile_path:
                return True
    return False

#todo: Supress similar matces within a radius
def get_best_tile(target, tiles_lowres, tiles_hires, tile_paths, chunk_indices):
    if not _tile_flat:    #simple dynamic programming
        for tile in tiles_lowres:
            tile_flat = np.array(tile).flatten() / 255.0
            _tile_flat.append(tile_flat)

    target_flat = target.flatten()/255.0

    lowest_diff = None
    best_index = None
    for index, tile_flat in enumerate(_tile_flat):
        diff = np.linalg.norm(target_flat-tile_flat)
        if (lowest_diff is None) or (diff < lowest_diff and not tile_within_radius(tile_paths[index], chunk_indices)):
            lowest_diff = diff
            best_index = index
            _placed_tiles[chunk_indices] = tile_paths[best_index]

    best_tile = tiles_hires[best_index]
    best_tile_img = Image.fromarray(best_tile)
    target_img = Image.fromarray(target).resize(best_tile.shape[0:2], Image.ANTIALIAS)
    blended_tile = np.array(Image.blend(best_tile_img, target_img, alpha=0.2))

    return blended_tile


def load_tiles(tiles_path, res_low, res_high):
    #return lowres, hires, paths
    print('Loading images and downsampling,', tiles_path)
    downsampled_tiles = {res_low: [], res_high: []}
    current_tiles_paths = list(glob.glob(tiles_path + '*.jpg'))

    for res in downsampled_tiles.keys():
        cache_path = tiles_path + 'cache_{}.pickle'.format(res)
        try:
            with open(cache_path, 'rb') as f:
                downsampled_tiles[res], cached_tiles_paths = pickle.load(f)
                if cached_tiles_paths != current_tiles_paths:
                    raise Exception('Tile cache for res {} outdated'.format(res))
        except Exception as e:
            print(e)
            print('Creating cache for res {}'.format(res))
            downsampled_tiles[res] = downsample_tiles_folder(tiles_path, res)
            with open(cache_path, 'wb') as f:
                pickle.dump((downsampled_tiles[res], current_tiles_paths), f)
        else:
            print('Loaded cached res {}'.format(res))

    return downsampled_tiles[res_low], downsampled_tiles[res_high], current_tiles_paths


def downsample_tiles_folder(tiles_path, res):
    tiles_downsampled = []
    n_files = len(glob.glob(tiles_path + '*.jpg'))
    for i, tile_img_path in enumerate(glob.glob(tiles_path + '*.jpg')):
        if i%25 == 0:
            print('\tdownsampling {}%'.format(float(i)/float(n_files)))
        tile_down = load_image(tile_img_path, size=res)
        tiles_downsampled.append(tile_down)
    print('\tdownsampling 100%')
    return tiles_downsampled



def load_image(path, size=None):
    img = Image.open(path)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    return np.array(img)


def show_image(img):
    plt.imshow(img)
    plt.show()


def create_disc_offsets(radius):
    for dm in range(-radius, radius+1, 1):
        for dn in range(-radius, radius+1, 1):
            if np.sqrt(dm**2 + dn**2) <= radius:
                _disc_offsets.append((dm, dn))

if __name__ == '__main__':
    long_args = ['tile_folder=', 'target_path=', 'save_path=', 'tile_size=', 'tile_resolution=', 'duplicate_radius=']
    opts, _ = getopt.getopt(sys.argv[1:], '', long_args)

    missing_args = ['--tile_folder', '--target_path', '--save_path', '--tile_size', '--tile_resolution', '--duplicate_radius']
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
    path_tiles = received_args['--tile_folder']
    path_target = received_args['--target_path']
    path_save = received_args['--save_path']
    tile_size = int(received_args['--tile_size'])
    tile_resolution = int(received_args['--tile_resolution'])
    duplicate_radius = int(received_args['--duplicate_radius'])


    create_disc_offsets(duplicate_radius)

    mosaic = create_mosaic(path_tiles, path_target, tile_size, tile_resolution, duplicate_radius)
    show_image(mosaic)
    mosaic_pil = Image.fromarray(mosaic)
    mosaic_pil.save(path_save)

