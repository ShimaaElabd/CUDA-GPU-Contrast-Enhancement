#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <ctime>

PGM_IMG gpu_contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;

    int histSize = 256;
    int *hist;
    int *d_hist;
    unsigned char * d_result;
    unsigned char * d_img_in;
    int *lut = (int *)malloc(sizeof(int)*256);
    int * d_lut;

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    hist = (int*)malloc(histSize*sizeof(int));	
    
    cudaMalloc(&d_result,result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_img_in,result.w * result.h * sizeof(unsigned char)); 
    cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMalloc(&d_lut,sizeof(int)*256);

    cudaMemcpy(d_result, result.img, result.w * result.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_in, img_in.img, result.w * result.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    gpu_histogram(hist, img_in.img, img_in.h * img_in.w, 256);

    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist[i++];
    }
    d = result.w*result.h - min;
    for(i = 0; i < 256; i ++){
        cdf += hist[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }
    cudaMemcpy(d_hist, hist, histSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut, sizeof(int)*256, cudaMemcpyHostToDevice);
    int numThreads = 1024;
    int blockSize = ((result.w*result.h)/1024)/3; 
    gpu_histogram_equalization<<<blockSize,numThreads>>>(d_result,d_img_in,d_hist,result.w*result.h, 256, blockSize * numThreads, d_lut);
    cudaMemcpy(result.img, d_result, result.w * result.h * sizeof(unsigned char), cudaMemcpyDeviceToHost);
   
    cudaFree(d_result);
    cudaFree(d_img_in);
    cudaFree(d_hist);
    cudaFree(d_lut);

    return result;
}

PPM_IMG gpu_contrast_enhancement_c_rgb(PPM_IMG img_in)
{

    PPM_IMG result;
    int histSize = 256;
    int *hist;
    int *d_hist;
    int *lut = (int *)malloc(sizeof(int)*256);
    int * d_lut;
    unsigned char * d_img_in_r;
    unsigned char * d_img_in_g;
    unsigned char * d_img_in_b;
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    hist = (int*)malloc(histSize*sizeof(int));

    // variables for gpu
    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;

    // set memory size
    cudaMalloc(&d_r, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_g, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_b, result.w * result.h * sizeof(unsigned char));
    cudaMalloc(&d_img_in_r, img_in.w * img_in.h * sizeof(unsigned char));
    cudaMalloc(&d_img_in_g, img_in.w * img_in.h * sizeof(unsigned char));
    cudaMalloc(&d_img_in_b, img_in.w * img_in.h * sizeof(unsigned char));
    cudaMalloc(&d_hist, histSize * sizeof(int));
    cudaMalloc(&d_lut,sizeof(int)*256);   

    // copy variables over to gpu variables
    cudaMemcpy(d_r, result.img_r, result.w * result.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, result.img_g, result.w * result.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, result.img_b, result.w * result.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_in_r, img_in.img_r, img_in.w * img_in.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_in_g, img_in.img_g, img_in.w * img_in.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_in_b, img_in.img_b, img_in.w * img_in.h * sizeof(unsigned char), cudaMemcpyHostToDevice);


    gpu_histogram(hist, img_in.img_r, img_in.h * img_in.w, 256);
 
    int i, cdf, min, d;
    
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist[i++];
    }
    d = result.w*result.h - min;

    for(i = 0; i < 256; i ++){
        cdf += hist[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }

    cudaMemcpy(d_hist, hist, histSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut, sizeof(int)*256, cudaMemcpyHostToDevice);
    int numThreads = 1024;
    int blockSize = ((result.w*result.h)/1024)/3; 
    gpu_histogram_equalization<<<blockSize,numThreads>>>(d_r,d_img_in_r,d_hist,result.w*result.h, 256,blockSize*numThreads,d_lut);

    gpu_histogram(hist, img_in.img_g, img_in.h * img_in.w, 256);

    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist[i++];
    }
    d = result.w*result.h - min;

    for(i = 0; i < 256; i ++){
        cdf += hist[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }

    cudaMemcpy(d_hist, hist, histSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut, sizeof(int)*256, cudaMemcpyHostToDevice);
    gpu_histogram_equalization<<<blockSize,numThreads>>>(d_g,d_img_in_g,d_hist,result.w*result.h, 256, blockSize * numThreads,d_lut);

    gpu_histogram(hist, img_in.img_b, img_in.h * img_in.w, 256);
    
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist[i++];
    }
    d = result.w*result.h - min;

    for(i = 0; i < 256; i ++){
        cdf += hist[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }

    cudaMemcpy(d_hist, hist, histSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut, sizeof(int)*256, cudaMemcpyHostToDevice);
    gpu_histogram_equalization<<<blockSize,numThreads>>>(d_b,d_img_in_b,d_hist,result.w*result.h, 256, blockSize * numThreads,d_lut);

    cudaMemcpy(result.img_r, d_r, result.w * result.h * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_g, d_g, result.w * result.h * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_b, d_b, result.w * result.h * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_hist);
    cudaFree(d_lut);

    return result;

}


PPM_IMG gpu_contrast_enhancement_c_yuv(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;

    using namespace std;
    
    unsigned char * y_equ; 
    unsigned char * d_y_equ;
    unsigned char * d_yuv_med;
    int histSize = 256;
    int *hist;
    int *d_hist;
    int *lut = (int *)malloc(sizeof(int)*256);
    int * d_lut;
    
    hist = (int*)malloc(histSize*sizeof(int));
    yuv_med = gpu_rgb2yuv(img_in);
    y_equ = (unsigned char *)malloc(yuv_med.h*yuv_med.w*sizeof(unsigned char));
    cudaMalloc(&d_y_equ, yuv_med.h*yuv_med.w*sizeof(unsigned char));
    cudaMalloc(&d_yuv_med, yuv_med.h*yuv_med.w*sizeof(unsigned char));
    cudaMalloc(&d_hist, histSize * sizeof(int));
    cudaMalloc(&d_lut,sizeof(int)*256);   
 
    cudaMemcpy(d_y_equ, y_equ, yuv_med.h*yuv_med.w*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yuv_med, yuv_med.img_y, yuv_med.h*yuv_med.w*sizeof(unsigned char), cudaMemcpyHostToDevice);

    gpu_histogram(hist, yuv_med.img_y, yuv_med.h * yuv_med.w, 256);

    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist[i++];
    }
    d = yuv_med.w*yuv_med.h - min;

    for(i = 0; i < 256; i ++){
        cdf += hist[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }


    cudaMemcpy(d_hist, hist, histSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut, sizeof(int)*256, cudaMemcpyHostToDevice);
    int numThreads = 1024;
    int blockSize = ((yuv_med.h * yuv_med.w)/1024)/3; 
    gpu_histogram_equalization<<<blockSize,numThreads>>>(d_y_equ,d_yuv_med,d_hist,yuv_med.h * yuv_med.w, 256,blockSize*numThreads,d_lut);

    cudaMemcpy(y_equ, d_y_equ, yuv_med.h*yuv_med.w*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
    result = gpu_yuv2rgb(yuv_med);
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);

    cudaFree(d_y_equ);
    cudaFree(d_yuv_med);
    cudaFree(d_hist);
    cudaFree(d_lut);
  
    return result;
}

PPM_IMG gpu_contrast_enhancement_c_hsl(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;
    
    unsigned char * l_equ;
    unsigned char * d_l_equ;
    unsigned char * d_hsl_med;
    int histSize = 256;
    int *hist;
    int *d_hist;
    int *lut = (int *)malloc(sizeof(int)*256);
    int * d_lut;

    hist = (int*)malloc(histSize*sizeof(int));	
    hsl_med = gpu_rgb2hsl(img_in);
    l_equ = (unsigned char *)malloc(hsl_med.height*hsl_med.width*sizeof(unsigned char));
    cudaMalloc(&d_l_equ, hsl_med.height*hsl_med.width*sizeof(unsigned char));
    cudaMalloc(&d_hsl_med, hsl_med.height*hsl_med.width*sizeof(unsigned char));
    cudaMalloc(&d_hist, histSize * sizeof(int));
    cudaMalloc(&d_lut,sizeof(int)*256);

    cudaMemcpy(d_l_equ, l_equ, hsl_med.height*hsl_med.width*sizeof(unsigned char), cudaMemcpyHostToDevice);
     cudaMemcpy(d_hsl_med, hsl_med.l, hsl_med.height*hsl_med.width*sizeof(unsigned char), cudaMemcpyHostToDevice);

    
    gpu_histogram(hist, hsl_med.l, hsl_med.height * hsl_med.width, 256);

    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist[i++];
    }
    d = hsl_med.width*hsl_med.height - min;

    for(i = 0; i < 256; i ++){
        cdf += hist[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }


    cudaMemcpy(d_hist, hist, histSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut, sizeof(int)*256, cudaMemcpyHostToDevice);
    int numThreads = 1024;
    int blockSize = ((hsl_med.width*hsl_med.height)/1024)/3; 
    gpu_histogram_equalization<<<blockSize,numThreads>>>(d_l_equ, d_hsl_med,d_hist,hsl_med.width*hsl_med.height, 256,blockSize*numThreads,d_lut);
    
    cudaMemcpy(l_equ, d_l_equ, hsl_med.height*hsl_med.width*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    free(hsl_med.l);
    hsl_med.l = l_equ;

    result = gpu_hsl2rgb(hsl_med);
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);

    cudaFree(d_l_equ);
    cudaFree(d_hsl_med);
    cudaFree(d_hist);
    cudaFree(d_lut);

    return result;
}


//Convert RGB to HSL, assume R,G,B in [0, 255]
//Output H, S in [0.0, 1.0] and L in [0, 255]
HSL_IMG gpu_rgb2hsl(PPM_IMG img_in)
{
    HSL_IMG img_out;

    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;
    float * d_h;
    float * d_s;
    unsigned char * d_l;

    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));
    
    cudaMalloc(&d_h, sizeof(float)*img_out.width*img_out.height);
    cudaMalloc(&d_s, sizeof(float)*img_out.width*img_out.height);
    cudaMalloc(&d_l, sizeof(unsigned char)*img_out.width*img_out.height);
    cudaMalloc(&d_r, sizeof(unsigned char)*img_in.w*img_in.h);
    cudaMalloc(&d_g, sizeof(unsigned char)*img_in.w*img_in.h);
    cudaMalloc(&d_b, sizeof(unsigned char)*img_in.w*img_in.h);
    
    cudaMemcpy(d_h, img_out.h, sizeof(float)*img_out.width*img_out.height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, img_out.s, sizeof(float)*img_out.width*img_out.height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, img_out.l, sizeof(unsigned char)*img_out.width*img_out.height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, img_in.img_r, sizeof(unsigned char)*img_in.w*img_in.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, img_in.img_g, sizeof(unsigned char)*img_in.w*img_in.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, img_in.img_b, sizeof(unsigned char)*img_in.w*img_in.h, cudaMemcpyHostToDevice);

    int numThreads = 1024;
    int blockSize = ((img_out.width*img_out.height)/1024)/3; 
    gpu_rgb2hslCal<<<blockSize,numThreads>>>(d_h, d_s, d_l, d_r, d_g, d_b, img_out.width*img_out.height, blockSize*numThreads);

    cudaMemcpy(img_out.h, d_h, sizeof(float)*img_out.width*img_out.height, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.s, d_s, sizeof(float)*img_out.width*img_out.height, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.l, d_l, sizeof(unsigned char)*img_out.width*img_out.height,
cudaMemcpyDeviceToHost);
    return img_out;
}

__global__ void gpu_rgb2hslCal(float * d_h , float * d_s ,unsigned char * d_l , unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, int size, int numOfThreads) {

    int i = 0;
    float H, S, L;
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    
    int start;
    int end;
    start = ((size/numOfThreads) * x);
    if(numOfThreads == 1) {
       end = (size/numOfThreads);
    }
    else {
       end = ((size/numOfThreads) * (x+1));
    }


     for(i = start; i < end; i ++){
        
        float var_r = ( (float)d_r[i]/255 );//Convert RGB to [0,1]
        float var_g = ( (float)d_g[i]/255 );
        float var_b = ( (float)d_b[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 )//This is a gray, no chroma...
        {
            H = 0;         
            S = 0;    
        }
        else                                    //Chromatic data...
        {
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;
            }
            else{       
                if( var_g == var_max ){
                    H = (1.0/3.0) + del_r - del_b;
                }
                else{
                        H = (2.0/3.0) + del_g - del_r;
                }   
            }
            
        }
        
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

        d_h[i] = H;
        d_s[i] = S;
        d_l[i] = (unsigned char)(L*255);
    }


}

__device__ float gpu_Hue_2_RGB( float v1, float v2, float vH )             //Function Hue_2_RGB
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}

//Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
//Output R,G,B in [0, 255]
PPM_IMG gpu_hsl2rgb(HSL_IMG img_in)
{
    PPM_IMG result;
    
    float * d_h;
    float * d_s;
    unsigned char * d_l;
    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;
    
    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    cudaMalloc(&d_r, sizeof(unsigned char)*result.w*result.h);
    cudaMalloc(&d_g, sizeof(unsigned char)*result.w*result.h);
    cudaMalloc(&d_b, sizeof(unsigned char)*result.w*result.h);
    cudaMalloc(&d_h, sizeof(float)*img_in.width*img_in.height);
    cudaMalloc(&d_s, sizeof(float)*img_in.width*img_in.height);
    cudaMalloc(&d_l, sizeof(unsigned char)*img_in.width*img_in.height);

    cudaMemcpy(d_r, result.img_r, sizeof(unsigned char)*result.w* result.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, result.img_g, sizeof(unsigned char)*result.w* result.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, result.img_b, sizeof(unsigned char)*result.w* result.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, img_in.h, sizeof(float)*img_in.width*img_in.height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, img_in.s, sizeof(float)*img_in.width*img_in.height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, img_in.l, sizeof(unsigned char)*img_in.width*img_in.height, cudaMemcpyHostToDevice);
    int numThreads = 1024;
    int blockSize = ((result.w*result.h)/1024)/3; 
    gpu_hsl2rgbCal<<<blockSize,numThreads>>>(d_h, d_s, d_l, d_r, d_g, d_b, result.w*result.h, blockSize*numThreads);

    cudaMemcpy(result.img_r, d_r, sizeof(unsigned char)*result.w*result.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_g, d_g, sizeof(unsigned char)*result.w*result.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.img_b, d_b, sizeof(unsigned char)*result.w*result.h, cudaMemcpyDeviceToHost);

    return result;
}

__global__ void gpu_hsl2rgbCal(float * d_h , float * d_s ,unsigned char * d_l , unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, int size, int numOfThreads) {

     int i = 0;

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    
    int start;
    int end;
    start = ((size/numOfThreads) * x);
    if(numOfThreads == 1) {
       end = (size/numOfThreads);
    }
    else {
       end = ((size/numOfThreads) * (x+1));
    }

    for(i = start; i < end; i ++){
        float H = d_h[i];
        float S = d_s[i];
        float L = d_l[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r,g,b;
        
        if ( S == 0 )
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {
            
            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            var_1 = 2 * L - var_2;
            r = 255 * gpu_Hue_2_RGB( var_1, var_2, H + (1.0f/3.0f) );
            g = 255 * gpu_Hue_2_RGB( var_1, var_2, H );
            b = 255 * gpu_Hue_2_RGB( var_1, var_2, H - (1.0f/3.0f) );
        }
        d_r[i] = r;
        d_g[i] = g;
        d_b[i] = b;
    }
}

//Convert RGB to YUV, all components in [0, 255]
YUV_IMG gpu_rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
   
    unsigned char * d_y;
    unsigned char * d_u;
    unsigned char * d_v;
    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
 
    cudaMalloc(&d_y, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc(&d_u, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc(&d_v, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc(&d_r, sizeof(unsigned char)*img_in.w*img_in.h);
    cudaMalloc(&d_g, sizeof(unsigned char)*img_in.w*img_in.h);
    cudaMalloc(&d_b, sizeof(unsigned char)*img_in.w*img_in.h);

     cudaMemcpy(d_y, img_out.img_y, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, img_out.img_u, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, img_out.img_v, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, img_in.img_r, sizeof(unsigned char)*img_in.w*img_in.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, img_in.img_g, sizeof(unsigned char)*img_in.w*img_in.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, img_in.img_b, sizeof(unsigned char)*img_in.w*img_in.h, cudaMemcpyHostToDevice);
    
    gpu_rbg2yuvCal<<<1024, 512>>>(d_y, d_u, d_v, d_r, d_g, d_b, img_out.w*img_out.h, 524288);
    cudaMemcpy(img_out.img_y, d_y, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_u, d_u, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_v, d_v, sizeof(unsigned char)*img_out.w*img_out.h,
cudaMemcpyDeviceToHost);
    return img_out;
}

__global__ void gpu_rbg2yuvCal(unsigned char * d_y , unsigned char * d_u ,unsigned char * d_v , unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, int size, int numOfThreads) {


    int i = 0;
    unsigned char r, g, b;
    unsigned char y, cb, cr;

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    
    int start;
    int end;
    start = ((size/numOfThreads) * x);
    if(numOfThreads == 1) {
       end = (size/numOfThreads);
    }
    else {
       end = ((size/numOfThreads) * (x+1));
    }


     for(i = start; i < end; i ++){
        r = d_r[i];
        g = d_g[i];
        b = d_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        d_y[i] = y;
        d_u[i] = cb;
        d_v[i] = cr;
    }
}

__device__ unsigned char gpu_clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

//Convert YUV to RGB, all components in [0, 255]
PPM_IMG gpu_yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;

    unsigned char * d_y;
    unsigned char * d_u;
    unsigned char * d_v;
    unsigned char * d_r;
    unsigned char * d_g;
    unsigned char * d_b;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    
    cudaMalloc(&d_r, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc(&d_g, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc(&d_b, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc(&d_y, sizeof(unsigned char)*img_in.w*img_in.h);
    cudaMalloc(&d_u, sizeof(unsigned char)*img_in.w*img_in.h);
    cudaMalloc(&d_v, sizeof(unsigned char)*img_in.w*img_in.h);

    cudaMemcpy(d_r, img_out.img_r, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, img_out.img_g, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, img_out.img_b, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, img_in.img_y, sizeof(unsigned char)*img_in.w*img_in.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, img_in.img_u, sizeof(unsigned char)*img_in.w*img_in.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, img_in.img_v, sizeof(unsigned char)*img_in.w*img_in.h, cudaMemcpyHostToDevice);
    int numThreads = 1024;
    int blockSize = ((img_out.w*img_out.h)/1024)/3; 
    gpu_yuv2rgbCal<<<blockSize,numThreads>>>(d_y, d_u, d_v, d_r, d_g, d_b, img_out.w*img_out.h, blockSize*numThreads);
    cudaMemcpy(img_out.img_r, d_r, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, d_g, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, d_b, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);
    return img_out;
}

__global__ void gpu_yuv2rgbCal(unsigned char * d_y , unsigned char * d_u ,unsigned char * d_v , unsigned char * r, unsigned char * g, unsigned char * b, int size, int numOfThreads) {
   
    int i = 0;
    int  rt,gt,bt;
    int y, cb, cr;

    int x = threadIdx.x + blockDim.x*blockIdx.x;
    
    int start;
    int end;
    start = ((size/numOfThreads) * x);
    if(numOfThreads == 1) {
       end = (size/numOfThreads);
    }
    else {
       end = ((size/numOfThreads) * (x+1));
    }

    for(i = start; i < end; i ++){
        y  = (int)d_y[i];
        cb = (int)d_u[i] - 128;
        cr = (int)d_v[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

        r[i] = gpu_clip_rgb(rt);
        g[i] = gpu_clip_rgb(gt);
        b[i] = gpu_clip_rgb(bt);

    }

}
