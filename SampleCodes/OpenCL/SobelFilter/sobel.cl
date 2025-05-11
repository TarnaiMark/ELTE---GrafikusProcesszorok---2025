
__constant sampler_t sampler =
      CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP
    | CLK_FILTER_NEAREST;

__kernel void sobel(write_only image2d_t output, read_only image2d_t input)
{
	const int x = get_global_id(0);
    const int y = get_global_id(1);

    float4 p[3*3];
    for(int dy = -1; dy <= 1; dy += 1)
    {
        for(int dx = -1; dx <= 1; dx += 1)
        {
            p[(dy+1) * 3 + (dx+1)] = read_imagef(input, sampler, (int2)(x+dx, y+dy));
        }
    }

    const float4 gradient_x = p[0*3+0] - p[0*3+2] + 2.0f * (p[1*3+0] - p[1*3+2]) + p[2*3+0] - p[2*3+2];
    const float4 gradient_y = p[0*3+0] - p[2*3+0] + 2.0f * (p[0*3+1] - p[2*3+1]) + p[0*3+2] - p[2*3+2];

    const float gradient = max(0.0f, min(1.0f, 0.25f * sqrt( dot(gradient_x, gradient_x) + dot(gradient_y, gradient_y) ) ) );

    write_imagef(output, (int2)(x, y), (float4)(gradient, gradient, gradient, 1.0f));
}