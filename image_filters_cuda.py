import numpy as np
from numba import cuda
from PIL import Image
import math

# grayscale filter
@cuda.jit
def grayscale_kernel(rgb_image_device, gray_image_device):
    """
    turn rgb image to grayscale on gpu
    """
    # get thread row and col
    i, j = cuda.grid(2)
    # check bounds
    if i < gray_image_device.shape[0] and j < gray_image_device.shape[1]:
        # get rgb values
        r = rgb_image_device[i, j, 0]
        g = rgb_image_device[i, j, 1]
        b = rgb_image_device[i, j, 2]
        # use standard grayscale formula
        gray_value = 0.2126 * r + 0.7152 * g + 0.0722 * b
        gray_image_device[i, j] = gray_value

def grayscale_image_cuda(input_image_path, output_image_path):
    """
    run grayscale filter
    """
    try:
        img = Image.open(input_image_path).convert('RGB')
        # normalize to [0, 1]
        host_rgb_image = np.array(img, dtype=np.float32) / 255.0 
    except FileNotFoundError:
        print(f"error: input image file not found at {input_image_path}")
        return
    except Exception as e:
        print(f"error loading image {input_image_path}: {e}")
        return

    height, width, channels = host_rgb_image.shape
    # copy rgb to gpu
    device_rgb_image = cuda.to_device(host_rgb_image)
    # make space for grayscale on gpu
    device_gray_image = cuda.device_array((height, width), dtype=np.float32)
    # set block and grid size
    threads_per_block_dim = (16, 16) 
    blocks_per_grid_x = math.ceil(width / threads_per_block_dim[1])
    blocks_per_grid_y = math.ceil(height / threads_per_block_dim[0])
    blocks_per_grid_dim = (blocks_per_grid_y, blocks_per_grid_x)
    # run kernel
    grayscale_kernel[blocks_per_grid_dim, threads_per_block_dim](device_rgb_image, device_gray_image)
    # copy result back
    host_gray_image = device_gray_image.copy_to_host()
    # convert to 8-bit for saving
    host_gray_image_8bit = np.clip(host_gray_image * 255.0, 0, 255).astype(np.uint8)
    try:
        Image.fromarray(host_gray_image_8bit, mode='L').save(output_image_path)
        print(f"grayscale image saved to {output_image_path}")
    except Exception as e:
        print(f"error saving image {output_image_path}: {e}")

# gaussian blur (global memory)
@cuda.jit
def gaussian_blur_kernel_global(input_image_device, output_image_device, kernel_device):
    """
    blur image using global memory
    """
    # get thread row and col
    i, j = cuda.grid(2) 
    img_height, img_width = input_image_device.shape
    kernel_height, kernel_width = kernel_device.shape
    kernel_radius_y = kernel_height // 2
    kernel_radius_x = kernel_width // 2
    # check bounds
    if i < img_height and j < img_width:
        accum_value = 0.0
        # loop over kernel
        for k_row in range(kernel_height):
            for k_col in range(kernel_width):
                neighbor_i = i - kernel_radius_y + k_row 
                neighbor_j = j - kernel_radius_x + k_col
                # check bounds for input
                if (neighbor_i >= 0 and neighbor_i < img_height and
                    neighbor_j >= 0 and neighbor_j < img_width):
                    accum_value += input_image_device[neighbor_i, neighbor_j] * kernel_device[k_row, k_col]
        output_image_device[i, j] = accum_value

def gaussian_blur_cuda_global(input_image_path, output_image_path, kernel_size=3):
    """
    run gaussian blur (global memory)
    """
    try:
        img = Image.open(input_image_path).convert('L')
        host_input_image = np.array(img, dtype=np.float32) / 255.0
    except FileNotFoundError:
        print(f"error: input image file not found at {input_image_path}")
        return
    except Exception as e:
        print(f"error loading image {input_image_path}: {e}")
        return
    height, width = host_input_image.shape
    # make gaussian kernel
    if kernel_size == 3:
        host_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    elif kernel_size == 5: 
        host_kernel = np.array([[1,  4,  7,  4,  1],
                                [4, 16, 26, 16,  4],
                                [7, 26, 41, 26,  7],
                                [4, 16, 26, 16,  4],
                                [1,  4,  7,  4,  1]], dtype=np.float32) / 273.0
    else: 
        print(f"unsupported kernel size {kernel_size}, using 3x3.")
        host_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
        kernel_size = 3
    # copy to gpu
    device_input_image = cuda.to_device(host_input_image)
    device_output_image = cuda.device_array_like(device_input_image)
    device_kernel = cuda.to_device(host_kernel)
    threads_per_block_dim = (16, 16)
    blocks_per_grid_x = math.ceil(width / threads_per_block_dim[1])
    blocks_per_grid_y = math.ceil(height / threads_per_block_dim[0])
    blocks_per_grid_dim = (blocks_per_grid_y, blocks_per_grid_x)
    gaussian_blur_kernel_global[blocks_per_grid_dim, threads_per_block_dim](
        device_input_image, device_output_image, device_kernel)
    host_output_image = device_output_image.copy_to_host()
    host_output_image_8bit = np.clip(host_output_image * 255.0, 0, 255).astype(np.uint8)
    try:
        Image.fromarray(host_output_image_8bit, mode='L').save(output_image_path)
        print(f"blurred image (global memory) saved to {output_image_path}")
    except Exception as e:
        print(f"error saving image {output_image_path}: {e}")

# gaussian blur (shared memory)
TILE_DIM_X = 16 
TILE_DIM_Y = 16
@cuda.jit
def gaussian_blur_kernel_shared(input_image_device, output_image_device, kernel_device):
    """
    blur image using shared memory
    """
    kernel_height, kernel_width = kernel_device.shape
    kernel_radius_y = kernel_height // 2
    kernel_radius_x = kernel_width // 2
    SHARED_TILE_HEIGHT = TILE_DIM_Y + 2 * kernel_radius_y 
    SHARED_TILE_WIDTH = TILE_DIM_X + 2 * kernel_radius_x
    s_tile = cuda.shared.array(shape=(SHARED_TILE_HEIGHT, SHARED_TILE_WIDTH), dtype=input_image_device.dtype)
    # get global thread index
    g_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    g_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # get local thread index
    l_y = cuda.threadIdx.y 
    l_x = cuda.threadIdx.x
    img_height, img_width = input_image_device.shape
    # find top-left of tile
    tile_origin_y_global = cuda.blockIdx.y * TILE_DIM_Y - kernel_radius_y
    tile_origin_x_global = cuda.blockIdx.x * TILE_DIM_X - kernel_radius_x
    # load pixels into shared memory
    for sy_offset in range(0, SHARED_TILE_HEIGHT, cuda.blockDim.y):
        s_load_y = l_y + sy_offset
        if s_load_y < SHARED_TILE_HEIGHT:
            g_load_y = tile_origin_y_global + s_load_y
            for sx_offset in range(0, SHARED_TILE_WIDTH, cuda.blockDim.x):
                s_load_x = l_x + sx_offset
                if s_load_x < SHARED_TILE_WIDTH:
                    g_load_x = tile_origin_x_global + s_load_x
                    if (g_load_y >= 0 and g_load_y < img_height and
                        g_load_x >= 0 and g_load_x < img_width):
                        s_tile[s_load_y, s_load_x] = input_image_device[g_load_y, g_load_x]
                    else:
                        s_tile[s_load_y, s_load_x] = 0.0
    # sync threads
    cuda.syncthreads()
    # do blur
    if g_y < img_height and g_x < img_width:
        accum_value = 0.0
        for k_row in range(kernel_height):
            for k_col in range(kernel_width):
                s_access_y = l_y + k_row 
                s_access_x = l_x + k_col
                accum_value += s_tile[s_access_y, s_access_x] * kernel_device[k_row, k_col]
        output_image_device[g_y, g_x] = accum_value

def gaussian_blur_cuda_shared(input_image_path, output_image_path, kernel_size=3):
    """
    run gaussian blur (shared memory)
    """
    try:
        img = Image.open(input_image_path).convert('L')
        host_input_image = np.array(img, dtype=np.float32) / 255.0
    except FileNotFoundError:
        print(f"error: input image file not found at {input_image_path}")
        return
    except Exception as e:
        print(f"error loading image {input_image_path}: {e}")
        return
    height, width = host_input_image.shape
    if kernel_size == 3:
        host_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    elif kernel_size == 5:
        host_kernel = np.array([[1,  4,  7,  4,  1],
                                [4, 16, 26, 16,  4],
                                [7, 26, 41, 26,  7],
                                [4, 16, 26, 16,  4],
                                [1,  4,  7,  4,  1]], dtype=np.float32) / 273.0
    else:
        print(f"unsupported kernel size {kernel_size}, using 3x3.")
        host_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
        kernel_size = 3
    device_input_image = cuda.to_device(host_input_image)
    device_output_image = cuda.device_array_like(device_input_image)
    device_kernel = cuda.to_device(host_kernel)
    threads_per_block_dim = (TILE_DIM_Y, TILE_DIM_X)
    blocks_per_grid_x = math.ceil(width / TILE_DIM_X)
    blocks_per_grid_y = math.ceil(height / TILE_DIM_Y)
    blocks_per_grid_dim = (blocks_per_grid_y, blocks_per_grid_x)
    gaussian_blur_kernel_shared[blocks_per_grid_dim, threads_per_block_dim](
        device_input_image, device_output_image, device_kernel)
    host_output_image = device_output_image.copy_to_host()
    host_output_image_8bit = np.clip(host_output_image * 255.0, 0, 255).astype(np.uint8)
    try:
        Image.fromarray(host_output_image_8bit, mode='L').save(output_image_path)
        print(f"blurred image (shared memory) saved to {output_image_path}")
    except Exception as e:
        print(f"error saving image {output_image_path}: {e}")

# sharpen filter
sharpen_kernel_convolution = gaussian_blur_kernel_global

def sharpen_image_cuda(input_image_path, output_image_path, kernel_type="standard", use_shared_memory=False):
    """
    run sharpen filter
    """
    try:
        img = Image.open(input_image_path).convert('L')
        host_input_image = np.array(img, dtype=np.float32) / 255.0
    except FileNotFoundError:
        print(f"error: input image file not found at {input_image_path}")
        return
    except Exception as e:
        print(f"error loading image {input_image_path}: {e}")
        return
    height, width = host_input_image.shape
    # make sharpen kernel
    if kernel_type == "standard":
        host_kernel = np.array([[ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]], dtype=np.float32)
    elif kernel_type == "stronger":
        host_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]], dtype=np.float32)
    else:
        print(f"unsupported sharpening kernel type '{kernel_type}', using standard.")
        host_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    device_input_image = cuda.to_device(host_input_image)
    device_output_image = cuda.device_array_like(device_input_image)
    device_kernel = cuda.to_device(host_kernel)
    if use_shared_memory:
        threads_per_block_dim = (TILE_DIM_Y, TILE_DIM_X)
        blocks_per_grid_x = math.ceil(width / TILE_DIM_X)
        blocks_per_grid_y = math.ceil(height / TILE_DIM_Y)
        blocks_per_grid_dim = (blocks_per_grid_y, blocks_per_grid_x)
        print("using shared memory kernel for sharpening.")
        gaussian_blur_kernel_shared[blocks_per_grid_dim, threads_per_block_dim](
            device_input_image, device_output_image, device_kernel)
    else:
        threads_per_block_dim = (16, 16)
        blocks_per_grid_x = math.ceil(width / threads_per_block_dim[1])
        blocks_per_grid_y = math.ceil(height / threads_per_block_dim[0])
        blocks_per_grid_dim = (blocks_per_grid_y, blocks_per_grid_x)
        print("using global memory kernel for sharpening.")
        sharpen_kernel_convolution[blocks_per_grid_dim, threads_per_block_dim](
            device_input_image, device_output_image, device_kernel)
    host_output_image = device_output_image.copy_to_host()
    # clamp to [0, 1]
    host_output_image_clamped = np.clip(host_output_image, 0.0, 1.0)
    host_output_image_8bit = (host_output_image_clamped * 255.0).astype(np.uint8)
    try:
        Image.fromarray(host_output_image_8bit, mode='L').save(output_image_path)
        print(f"sharpened image saved to {output_image_path}")
    except Exception as e:
        print(f"error saving image {output_image_path}: {e}")

# test all filters
if __name__ == '__main__':
    print("testing numba cuda image filters")
    # make dummy images if needed
    base_width, base_height = 256, 256
    input_rgb_path = 'input_rgb_numba.png'
    try:
        Image.open(input_rgb_path)
    except FileNotFoundError:
        print(f"creating dummy rgb image: {input_rgb_path}")
        Image.new('RGB', (base_width, base_height), color='red').save(input_rgb_path)
    output_grayscale_path = 'output_grayscale_numba.png'
    # test grayscale
    print("\ntesting grayscale conversion...")
    grayscale_image_cuda(input_rgb_path, output_grayscale_path)
    # check if grayscale output exists
    try:
        Image.open(output_grayscale_path)
        can_proceed = True
    except FileNotFoundError:
        print(f"grayscale output {output_grayscale_path} not found. skipping numba tests.")
        can_proceed = False
    except Exception as e:
        print(f"error opening grayscale output {output_grayscale_path}: {e}. skipping numba tests.")
        can_proceed = False
    if can_proceed:
        # test blur (global)
        print("\ntesting gaussian blur (global memory)...")
        gaussian_blur_cuda_global(output_grayscale_path, 'output_blur_global_numba.png', kernel_size=3)
        gaussian_blur_cuda_global(output_grayscale_path, 'output_blur_global_5x5_numba.png', kernel_size=5)
        # test blur (shared)
        print("\ntesting gaussian blur (shared memory)...")
        gaussian_blur_cuda_shared(output_grayscale_path, 'output_blur_shared_numba.png', kernel_size=3)
        gaussian_blur_cuda_shared(output_grayscale_path, 'output_blur_shared_5x5_numba.png', kernel_size=5)
        # test sharpen (global)
        print("\ntesting sharpening (global memory)...")
        sharpen_image_cuda(output_grayscale_path, 'output_sharpened_global_numba.png', kernel_type="standard", use_shared_memory=False)
        print("\ntesting sharpening (shared memory)...")
        sharpen_image_cuda(output_grayscale_path, 'output_sharpened_shared_numba.png', kernel_type="standard", use_shared_memory=True)
    print("\nnumba cuda image filter tests complete")