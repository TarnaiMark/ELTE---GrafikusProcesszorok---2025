#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "helper_math.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void sobel(cudaSurfaceObject_t output, cudaTextureObject_t input)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    float4 p[3*3];
    for(int dy = -1; dy <= 1; dy += 1)
    {
        for(int dx = -1; dx <= 1; dx += 1)
        {
            p[(dy+1) * 3 + (dx+1)] = tex2D<float4>(input, x+dx, y+dy);
        }
    }

    const float4 gradient_x = p[0*3+0] - p[0*3+2] + 2.0f * (p[1*3+0] - p[1*3+2]) + p[2*3+0] - p[2*3+2];
    const float4 gradient_y = p[0*3+0] - p[2*3+0] + 2.0f * (p[0*3+1] - p[2*3+1]) + p[0*3+2] - p[2*3+2];

    const float gradient = max(0.0f, min(1.0f, 0.25f * sqrt( dot(gradient_x, gradient_x) + dot(gradient_y, gradient_y) ) ) );
    surf2Dwrite(float4{gradient, gradient, gradient, 1.0f}, output, x * sizeof(float4), y, cudaSurfaceBoundaryMode::cudaBoundaryModeZero);
}

struct rawcolor{ unsigned char r, g, b; };

int main()
{
    cudaError_t err = cudaSuccess;
    
    // Using the implicitely selected first cuda device:
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if(err != cudaSuccess){ std::cout << "Error getting device properties for device 0: " << cudaGetErrorString(err) << "\n"; return -1; }
    std::cout << "Selected device name: " << prop.name << "\n";
    
    // Actual program logic:
    // Read input image:
    int w = 0; // width
    int h = 0; // height
    int ch = 0; // number of components

    std::string input_filename = "Valve_original.png";
    rawcolor* data0 = reinterpret_cast<rawcolor*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 3 /* we expect 3 components */));
    if(!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        std::cout << "Image (" << input_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }

    std::vector<float4> input (w*h);
    std::vector<float4> output(w*h);

    std::transform(data0, data0+w*h, input.begin(), [](rawcolor c){ return float4{c.r/255.0f, c.g/255.0f, c.b/255.0f, 1.0f}; } );
    
    // Create cuda texture and surface objects:
    // Channel layout of data:
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
       
    // Allocate data:
    cudaArray* arr_input = nullptr;
    cudaArray* arr_output = nullptr;

    err = cudaMallocArray(&arr_input, &channelDesc, w, h);
    if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory arr_input: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaMallocArray(&arr_output, &channelDesc, w, h);
    if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory arr_output: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Upload data to device:
    err = cudaMemcpyToArray(arr_input,  0, 0, input.data(), w*h*sizeof(float4), cudaMemcpyHostToDevice);
    if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }
        
    // Specify texture resource description:
    cudaResourceDesc resdescInput{};
    resdescInput.resType = cudaResourceTypeArray;
    resdescInput.res.array.array = arr_input;

    // Specify texture description:
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0]   = cudaAddressModeBorder;
    texDesc.addressMode[1]   = cudaAddressModeBorder;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texture = 0;
    err = cudaCreateTextureObject(&texture,  &resdescInput,  &texDesc, nullptr);
    if( err != cudaSuccess){ std::cout << "Error creating texture object: " << cudaGetErrorString(err) << "\n"; return -1; }
        
    // Create the surface object that will hold the output:
    cudaSurfaceObject_t surface;
    // Specify surface resource description:
    cudaResourceDesc resdescOutput{};
    resdescOutput.resType = cudaResourceTypeArray;
    resdescOutput.res.array.array = arr_output;

    err = cudaCreateSurfaceObject(&surface, &resdescOutput);
    if( err != cudaSuccess){ std::cout << "Error creating surface object: " << cudaGetErrorString(err) << "\n"; return -1; }
    
    // Launch kernel:
    {
        const int block = 16;
        dim3 grid_size( static_cast<int>( std::ceil(w*1.0f/block) ), static_cast<int>( std::ceil(w*1.0f/block) ) );
	    dim3 block_size( block, block );
	    sobel<<<grid_size, block_size>>>(surface, texture);

	    err = cudaGetLastError();
	    if(err != cudaSuccess){ std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
    }

    // Copy back results (implicitely synchronizes on the default stream that we are using):
    err = cudaMemcpyFromArray(output.data(), arr_output, 0, 0, w*h*sizeof(float4), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){ std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

    // Write out image:
    {
        std::vector<rawcolor> tmp(w*h);
        std::transform(output.cbegin(), output.cend(), tmp.begin(),
            [](float4 c){ return rawcolor{  (unsigned char)(c.x*255.0f),
                                            (unsigned char)(c.y*255.0f),
                                            (unsigned char)(c.z*255.0f) }; } );

        const std::string output_filename = "result.jpg";
        int res = stbi_write_jpg(output_filename.c_str(), w, h, ch, tmp.data(), 100);
        if(res == 0)
        {
            std::cout << "Error writing output to file " << output_filename << "\n";
        }else{ std::cout << "Output written to file " << output_filename << "\n"; }
    }

    // Clean-up:
	err = cudaDestroySurfaceObject(surface);
    if(err != cudaSuccess){ std::cout << "Error destroying surface object: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaDestroyTextureObject(texture);
    if(err != cudaSuccess){ std::cout << "Error destroying texture object: " << cudaGetErrorString(err) << "\n"; return -1; }

	err = cudaFreeArray( arr_output );
	if(err != cudaSuccess){ std::cout << "Error freeing array arr_output: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFreeArray( arr_input );
	if(err != cudaSuccess){ std::cout << "Error freeing array arr_input: " << cudaGetErrorString(err) << "\n"; return -1; }

    stbi_image_free(data0);

	return 0;
}