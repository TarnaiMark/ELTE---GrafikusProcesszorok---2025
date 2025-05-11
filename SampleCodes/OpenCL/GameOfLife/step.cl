__kernel void step( __global uchar4* pixels, __global unsigned int* output, __global unsigned int* input, int w, int h )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    int self = input[y*w+x];
    int living = 0;

    for(int dy = -1; dy <= 1; dy += 1)
    {
        const int yr = y + dy;
        if(yr < 0 || yr >= h){ continue; }
        for(int dx = -1; dx <= 1; dx += 1)
        {
            const int xr = x + dx;
            if(xr < 0 || xr >= w || (dy == 0 && dx == 0)){ continue; }
            int read = input[yr*w+xr];
            if(read == 1)
            {
                living += 1;
            }
        }
    }

    // update rules:
    if     (self == 1 && living < 2){ self = 0; }
    else if(self == 1 && (living == 2 || living == 3)){ self = 1; }
    else if(self == 1 && living > 3){ self = 0; }
    else if(self == 0 && living == 3){ self = 1; }

    // store:
    output[y*w+x] = self;

    uchar4 px;
    if(self)
    {
        px = (uchar4)(0, 128, 0, 255); // living color
    }
    else
    {
        px = (uchar4)(224, 224, 224, 255); // dead color
    }
    pixels[y*w+x] = px;
}