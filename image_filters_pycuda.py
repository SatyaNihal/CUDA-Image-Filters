import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from PIL import Image
import math

def grayscale_image_pycuda(input_image_path, output_image_path):
    """
    grayscale with pycuda
    """
    # cuda kernel as string
    cuda_kernel_code = """
    __global__ void grayscale_kernel_pycuda(const float* rgb_image, float* gray_image, int width, int height)
    {
        // get thread x and y
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;

        // check bounds
        if (i < height && j < width)
        {
            // index for grayscale output
            int gray_idx = i * width + j;
            // index for rgb input (3 channels)
            int rgb_base_idx = (i * width + j) * 3;

            float r = rgb_image[rgb_base_idx + 0];
            float g = rgb_image[rgb_base_idx + 1];
            float b = rgb_image[rgb_base_idx + 2];

            // use standard grayscale formula
            gray_image[gray_idx] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }
    }
    """
    try:
        img = Image.open(input_image_path).convert('RGB')
        # normalize to [0,1] float32
        host_rgb_image = np.array(img, dtype=np.float32) / 255.0
    except FileNotFoundError:
        print(f"Error: Input image file not found at {input_image_path}")
        return
    except Exception as e:
        print(f"Error loading image {input_image_path}: {e}")
        return

    height, width, channels = host_rgb_image.shape
    if channels != 3:
        print("pycuda needs a 3-channel rgb image.")
        return
        
    # make space for grayscale output
    host_gray_image = np.empty((height, width), dtype=np.float32)

    # copy rgb image to gpu
    device_rgb_image = drv.mem_alloc(host_rgb_image.nbytes)
    drv.memcpy_htod(device_rgb_image, host_rgb_image)
    # make space for grayscale on gpu
    device_gray_image = drv.mem_alloc(host_gray_image.nbytes)

    # compile kernel
    try:
        module = SourceModule(cuda_kernel_code)
        grayscale_func_pycuda = module.get_function("grayscale_kernel_pycuda")
    except drv.CompileError as e:
        print("pycuda kernel compile failed:")
        print(e)
        return

    # set block and grid size
    threads_per_block_dim = (16, 16, 1)
    grid_dim_x = math.ceil(width / threads_per_block_dim[0])
    grid_dim_y = math.ceil(height / threads_per_block_dim[1])
    grid_dim = (grid_dim_x, grid_dim_y)

    # run kernel
    grayscale_func_pycuda(device_rgb_image, device_gray_image, 
                          np.int32(width), np.int32(height),
                          block=threads_per_block_dim, grid=grid_dim)
    
    # copy result back
    drv.memcpy_dtoh(host_gray_image, device_gray_image)

    # convert to 8-bit for saving
    host_gray_image_8bit = np.clip(host_gray_image * 255.0, 0, 255).astype(np.uint8)
    
    try:
        Image.fromarray(host_gray_image_8bit, mode='L').save(output_image_path)
        print(f"grayscale image (pycuda) saved to {output_image_path}")
    except Exception as e:
        print(f"error saving image {output_image_path}: {e}")
        
    # gpu memory is managed by pycuda.autoinit
    # can free manually if needed
    # device_rgb_image.free()
    # device_gray_image.free()

if __name__ == '__main__':
    print("testing pycuda image filter (grayscale)")
    
    base_width, base_height = 256, 256
    input_rgb_pycuda_path = 'input_rgb_pycuda.png'
    
    try:
        Image.open(input_rgb_pycuda_path)
    except FileNotFoundError:
        print(f"creating dummy rgb image for pycuda: {input_rgb_pycuda_path}")
        Image.new('RGB', (base_width, base_height), color='blue').save(input_rgb_pycuda_path)
    
    output_grayscale_pycuda_path = 'output_grayscale_pycuda.png'
    grayscale_image_pycuda(input_rgb_pycuda_path, output_grayscale_pycuda_path)
    
    print("\npycuda grayscale test complete")