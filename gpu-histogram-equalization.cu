#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


void gpu_histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

__global__ void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin, int numOfThreads, int * lut){


    int i = 0;
    int x = threadIdx.x + blockDim.x*blockIdx.x;

    int start;
    int end;
    //hist_in[x%256] = x;
    /* Get the result image */
    if(x >= img_size) {
       return;
    }
    start = ((img_size/numOfThreads) * x);
    if(numOfThreads == 1) {
       end = (img_size/numOfThreads);
    }
    else {
       end = ((img_size/numOfThreads) * (x+1));
    }
    for(i = start; i < end; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}


