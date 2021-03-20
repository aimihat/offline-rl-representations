import argparse
import glob
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import rasterfairy
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def grid_tsne(tsne, observations):
    # nx * ny = 1000, the number of images
    nx = 40
    ny = 25

    # assign to grid
    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))

    tile_width = 84
    tile_height = 84

    full_width = tile_width * nx
    full_height = tile_height * ny
    aspect_ratio = float(tile_width) / tile_height

    grid_image = Image.new("L", (full_width, full_height))

    for img, grid_pos in zip(observations[..., 0], grid_assignment[0]):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        tile = Image.fromarray(img)
        tile_ar = (
            float(tile.width) / tile.height
        )  # center-crop the tile to match aspect_ratio
        if tile_ar > aspect_ratio:
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop(
                (margin, 0, margin + aspect_ratio * tile.height, tile.height)
            )
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop(
                (0, margin, tile.width, margin + float(tile.width) / aspect_ratio)
            )
        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    plt.figure(figsize=(32, 24))
    plt.imshow(grid_image)
    plt.savefig("grid_image.png")


def main(args):
    # Load (abstraction, observation) pairs
    with open(args.load_path, "rb") as f:
        data = pickle.load(f)
    abstractions = data["abstractions"]
    observations = data["observations"]
    # Reduce abstractions to 300 dimensions with PCA
    print("PCA")
    pca = PCA(n_components=300)
    abstractions_projection = pca.fit_transform(abstractions)
    print("T-SNE")
    # Obtain 2D T-SNE Embedding
    tsne_dimensions = 2
    tsne = TSNE(n_components=tsne_dimensions, verbose=5).fit_transform(
        abstractions_projection
    )
    # normalize embedding [0,1]
    tx, ty = tsne[:, 0], tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 4000
    height = 3000
    max_dim = 300

    full_image = Image.new("L", (width, height))
    for img, x, y in zip(observations[..., 0], tx, ty):
        tile = Image.fromarray(img)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize(
            (int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS
        )
        full_image.paste(
            tile, (int((width - max_dim) * x), int((height - max_dim) * y))
        )  # , mask=tile.convert('RGBA'))

    plt.figure(figsize=(32, 24))
    plt.imshow(full_image)
    plt.savefig("full_image.png")

    # grid tsne
    grid_tsne(tsne, observations)

    # # Delete contents
    # for f in glob.glob(args.images_path + "*"):
    #     os.remove(f)
    # # save observations as images + their T-SNE to JSON (currently first frame of each stack)
    # jsondata = []
    # for i, obs in enumerate(observations):
    #     im = Image.fromarray(obs[..., 0])
    #     imgpath = f"{args.images_path}obs_{str(i)}.jpeg"
    #     im.save(imgpath)
    #     point = [
    #         float(
    #             (tsne[i, k] - np.min(tsne[:, k]))
    #             / (np.max(tsne[:, k]) - np.min(tsne[:, k]))
    #         )
    #         for k in range(tsne_dimensions)
    #     ]
    #     jsondata.append({"path": imgpath, "point": point})
    # with open(
    #     f"{args.images_path}points.json",
    #     "w",
    # ) as outfile:
    #     json.dump(jsondata, outfile)
    # # save data to json
    # data = []
    # for i,f in enumerate(images):
    #     point = [float((tsne[i,k] - np.min(tsne[:,k]))/(np.max(tsne[:,k]) - np.min(tsne[:,k]))) for k in range(tsne_dimensions) ]
    #     data.append({"path":os.path.abspath(join(images_path,images[i])), "point":point})
    # with open(output_path, 'w') as outfile:
    #     json.dump(data, outfile)

    # import pdb
    # pdb.set_trace()
    # plt.scatter(abstractions_embedded[:,0],abstractions_embedded[:,1])
    # plt.savefig('tsne.png')
    # # Plot and sample random images to show.
    # # Image t-SNE viewer


if __name__ == "__main__":
    # Parsing args
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str)
    parser.add_argument("--images-path", type=str)
    args = parser.parse_args()
    assert args.images_path[-1] == "/"
    main(args)
