__constant int imageSize = 900;
__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;
__constant int2 kernel_size = (int2)(3 ,3);


__kernel void Laplacian( __read_only image2d_t input, __write_only image2d_t output, __global float* filter) {
    int2 uv = (int2) (get_global_id(0), get_global_id(1));

    uint4 color = read_imageui(input, image_sampler, uv);
    float4 finalColor = 0;

    for (int i = 0; i < kernel_size.y; i++) {
        for (int j = 0; j < kernel_size.y; j++) {
            float4 fColor = convert_float4(read_imageui(input, image_sampler, uv + (int2)(i-1, j-1)));
            float4 filterColor = filter[i * kernel_size.y + j];

            finalColor += fColor * filterColor;
        }
    }
    
    write_imageui(output, uv, convert_uint4(finalColor));
}