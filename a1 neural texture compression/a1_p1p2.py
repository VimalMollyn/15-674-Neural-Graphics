# import torch
from PIL import Image
import numpy as np
import math
from pathlib import Path

# P1, assumes a global texture
def sample_p1(u, v):
    # get the top left coordinates, make sure that they are in bounds of the uv map, here we just clamp to 1
    x_top_left = math.floor(u * texture_width)
    y_top_left = math.floor(v * texture_height)

    s = u * texture_width - x_top_left
    t = v * texture_height - y_top_left

    # clamp values at the edge
    x00 = np.clip(x_top_left, 0, texture_width-1)
    y00 = np.clip(y_top_left, 0, texture_height-1)

    x10 = np.clip(x_top_left + 1, 0, texture_width-1)
    y10 = np.clip(y_top_left, 0, texture_height-1)

    x01 = np.clip(x_top_left, 0, texture_width-1)
    y01 = np.clip(y_top_left + 1, 0, texture_height-1)

    x11 = np.clip(x_top_left + 1, 0, texture_width-1)
    y11 = np.clip(y_top_left + 1, 0, texture_height-1)

    c00 = texture[y00, x00]
    c10 = texture[y10, x10]
    c01 = texture[y01, x01]
    c11 = texture[y11, x11]

    color = (1-s)*(1-t)*c00 + s*(1-t)*c10 + (1-s)*t*c01 + s*t*c11
    return color

def to_rgb565(c):
    levels = np.array([31, 63, 31])                # 5, 6, 5 bits for R, G, B
    q      = np.round(c * levels)           # integer 565 code
    c_hat  = q / levels                  # back to [0, 1]
    return c_hat

def get_psnr(original, compressed):
    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)

# P2

def s3tc_compress(texture_path):
    texture = np.asarray(Image.open(texture_path))

    texture_height, texture_width, _ = texture.shape
    # divide into blocks

    # block size 4x4, assuming texture_width and height divide by 4
    # texture_blocks = texture.reshape(texture_height//4, 4, texture_width//4, 4, 3).swapaxes(1, 2).reshape(-1, 4, 4, 3)

    # compressed_texture = np.zeros_like(texture)

    compressed_texture_blocks = []
    compressed_texture_palettes = []

    # to be vectorized later
    for block_row in range(texture_height // 4):
        for block_col in range(texture_width // 4):
            # print(block.shape) # 4x4x3
            # convert colors to 0,1
            block = texture[block_row * 4: block_row*4 + 4, block_col*4:block_col*4 + 4]
            block = block / 255

            # choosing c0 and c1 based on line assumption
            c0 = to_rgb565(np.max(block.reshape(-1, 3), axis=0))
            c1 = to_rgb565(np.min(block.reshape(-1, 3), axis=0))

            # we choose c0 c1 such that c0 > c1 so always four colors
            c2 = (2*c0 + c1) / 3
            c3 = (c0 + 2*c1) / 3

            block_palette = np.array([c0, c1, c2, c3])

            # now assign new texture values
            # compressed_block = np.zeros((4, 4, 3))
            compressed_block = np.zeros((4, 4, 1))

            # for each texel in block, find the cloeset color
            for i in range(4):
                for j in range(4):
                    texel = block[i,j]
                    color_index = np.argmin(np.linalg.norm(texel - block_palette, axis=1))
                    # color = block_palette[color_index] * 255
                    compressed_block[i,j] = color_index

            compressed_texture_blocks.append(compressed_block)
            compressed_texture_palettes.append(block_palette)

    compressed_texture_blocks = np.array(compressed_texture_blocks).reshape(texture_height//4, texture_width//4, 4, 4, 1)
    compressed_texture_palettes = np.array(compressed_texture_palettes).reshape(texture_height//4, texture_width//4, 4, 3)

    def sample(u, v):
        # get the top left coordinates, make sure that they are in bounds of the uv map, here we just clamp to 1
        x_top_left = math.floor(u * texture_width)
        y_top_left = math.floor(v * texture_height)

        s = u * texture_width - x_top_left
        t = v * texture_height - y_top_left

        # clamp values at the edge
        # in block coordinates
        x00 = np.clip(x_top_left, 0, texture_width-1)
        y00 = np.clip(y_top_left, 0, texture_height-1)

        x10 = np.clip(x_top_left + 1, 0, texture_width-1)
        y10 = np.clip(y_top_left, 0, texture_height-1)

        x01 = np.clip(x_top_left, 0, texture_width-1)
        y01 = np.clip(y_top_left + 1, 0, texture_height-1)

        x11 = np.clip(x_top_left + 1, 0, texture_width-1)
        y11 = np.clip(y_top_left + 1, 0, texture_height-1)

        def block_coord(y, x):
            block_x = x // 4
            block_y = y // 4
            return block_y, block_x
        
        def local_coord(y, x):
            local_x = x % 4
            local_y = y % 4
            return local_y, local_x

        def get_color(y, x):
            b_y, b_x = block_coord(y, x) 
            block = compressed_texture_blocks[b_y, b_x].squeeze() # 4x4
            palette = compressed_texture_palettes[b_y, b_x] # 4x3
            # print(local_coord(y, x))
            l_y, l_x = local_coord(y, x)
            color_index = block[l_y, l_x].astype(int) # 4, 
            # print(block.shape, palette.shape, color_index.shape)
            return palette[color_index] * 255.0

        c00 = get_color(y00, x00)
        c10 = get_color(y10, x10)
        c01 = get_color(y01, x01)
        c11 = get_color(y11, x11)

        color = (1-s)*(1-t)*c00 + s*(1-t)*c10 + (1-s)*t*c01 + s*t*c11
        return color

    uv_coords = np.stack(np.meshgrid(np.arange(texture_width), np.arange(texture_height)), axis=-1).reshape(-1, 2) / np.array([texture_width, texture_height])
    # decode
    # sample all points in uv space and assign colors based on the compressed texture
    compressed_texture = np.zeros_like(texture)
    for uv in uv_coords:
        u = uv[0]
        v = uv[1]
        color = sample(u, v)
        compressed_texture[int(v * texture_height), int(u * texture_width)] = color


    # compressed_texture = np.round(np.stack(compressed_texture_blocks).reshape(texture_height // 4, texture_width//4, 4, 4, 3).swapaxes(1,2).reshape(texture_height, texture_width, 3)).astype(np.uint8)
    compressed_texture = np.round(compressed_texture).astype(np.uint8)
    save_dir = texture_path.parent.parent / "s3tc_compressed"
    save_dir.mkdir(exist_ok=True)
    Image.fromarray(compressed_texture).save(save_dir / texture_path.name)

    # compute psnr
    psnr = get_psnr(texture, compressed_texture) 

    return psnr


# texture_names = ["gradient.png", "bricks.png", "clouds.png"]

# for P8
texture_names = ["vimal.png", "body.png", "many_freqs.png"]

for name in texture_names:
    texture_path = Path("./textures") / name
    psnr = s3tc_compress(texture_path)
    print(texture_path, psnr)
