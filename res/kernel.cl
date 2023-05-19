__constant int imageSize = 900;
__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;


__kernel void Laplacian( __read_only image2d_t input, __write_only image2d_t output)
{
    int2 uv = (int2) (get_global_id(0), get_global_id(1));

    uint4 color = read_imageui(input, image_sampler, uv);
    int gray = (color.r + color.g + color.b) / 3;
    
    write_imageui(output, uv, (uint4)(gray, gray, gray, color.w));
}