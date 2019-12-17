import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import numpy as np

mod = SourceModule(
    """
    surface<void, 2> surf_image_2D;

    __global__ void write_surf_data()
    {
        int w = blockIdx.x * blockDim.x + threadIdx.x;
        int h = blockIdx.y * blockDim.y + threadIdx.y;

        float data = h * 4 + w;
        surf2Dwrite(data, surf_image_2D, w * 4, h);
    }
    """
)

desc = cuda.ArrayDescriptor3D()
desc.width = 4
desc.height = 4
desc.num_channels = 1
desc.format = cuda.dtype_to_array_format(np.float32)
desc.flags |= cuda.array3d_flags.SURFACE_LDST
x_array = cuda.Array(desc)

surf_image_2D = mod.get_surfref("surf_image_2D")
write_surf_data = mod.get_function("write_surf_data")

surf_image_2D.set_array(x_array)
write_surf_data(block=(4, 4, 1), grid=(1, 1))
x = np.zeros(shape=(4, 4), dtype=np.float32)

copy = cuda.Memcpy3D()
copy.set_src_array(x_array)
copy.set_dst_host(x)
copy.width_in_bytes = 4 * 4
copy.height = 4
copy.depth = 1
copy()
print(x)