#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include "util.h"
#include "MBD_distance_2d.h"
#include <queue>
#include <stdio.h>
#include<stdlib.h> 

using namespace std;





void MBD_waterflow(const unsigned char * img, const unsigned char * seeds, unsigned char * distance, 
                              int height, int width)
{
    int * state = new int[height * width];
    unsigned char * min_image = new unsigned char[height * width];
    unsigned char * max_image = new unsigned char[height * width];

    vector<queue<Point2D> > Q(256);

    // point state: 0--acceptd, 1--temporary, 2--far away
    // get initial accepted set and far away set

    unsigned char init_dis;
    int init_state;

    for(int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            Point2D p;
            p.h = h;
            p.w = w;
            unsigned char seed_type = get_pixel<unsigned char>(seeds, height, width, h, w);
            unsigned char img_value = get_pixel<unsigned char>(img, height, width, h, w);

            if(seed_type > 100){
                init_dis = 0;
                init_state = 1;
                Q[init_dis].push(p);
                set_pixel<unsigned char>(distance, height, width, h, w, init_dis);
                set_pixel<int>(state, height, width, h, w, init_state);
                set_pixel<unsigned char>(min_image, height, width, h, w, img_value);
                set_pixel<unsigned char>(max_image, height, width, h, w, img_value);                    
            }
            else{
                init_dis = 255;
                init_state = 0;
                set_pixel<unsigned char>(distance, height, width, h, w, init_dis);
                set_pixel<int>(state, height, width, h, w, init_state);  
                set_pixel<unsigned char>(min_image, height, width, h, w, img_value);
                set_pixel<unsigned char>(max_image, height, width, h, w, img_value);                                        
            }
        }
    }


    int dh[4] = { 1 ,-1 , 0, 0};
    int dw[4] = { 0 , 0 , 1,-1};

    // Proceed the propagation from the marker to all pixels in the image
    for (int lvl = 0; lvl < 256; lvl++)
    {

        while (!Q[lvl].empty())
        {
            Point2D p = Q[lvl].front();
            Q[lvl].pop();

            int state_value = get_pixel<int>(state, height, width, p.h, p.w);
            if (state_value == 2)
                continue;

            set_pixel<int>(state, height, width, p.h, p.w, 2);


            for (int n1 = 0 ; n1 < 4 ; n1++)
            {
                int tmp_h  = p.h + dh[n1];
                int tmp_w  = p.w + dw[n1];

                if (tmp_h >= 0 and tmp_h < height and tmp_w >= 0 and tmp_w < width)
                {
                    Point2D r;
                    r.h = tmp_h;
                    r.w = tmp_w;

                    unsigned char temp_r = get_pixel<unsigned char>(distance, height, width,  r.h, r.w);
                    unsigned char temp_p = get_pixel<unsigned char>(distance, height, width,  p.h, p.w);

                    state_value = get_pixel<int>(state, height, width, r.h, r.w);

                    if (state_value == 1 && temp_r> temp_p)
                    {
                        unsigned char min_image_value = get_pixel<unsigned char>(min_image, height, width, p.h, p.w);
                        unsigned char max_image_value = get_pixel<unsigned char>(max_image, height, width, p.h, p.w);

                        set_pixel<unsigned char>(min_image, height, width, r.h, r.w, min_image_value);
                        set_pixel<unsigned char>(max_image, height, width, r.h, r.w, max_image_value);

                        unsigned char image_value = get_pixel<unsigned char>(img, height, width, r.h, r.w);

                        if (image_value < min_image_value)
                            set_pixel<unsigned char>(min_image, height, width, r.h, r.w, image_value);
                        if (image_value > max_image_value)
                            set_pixel<unsigned char>(max_image, height, width, r.h, r.w, image_value);

                        temp_r = get_pixel<unsigned char>(distance, height, width, r.h, r.w);

                        min_image_value = get_pixel<unsigned char>(min_image, height, width, r.h, r.w);
                        max_image_value = get_pixel<unsigned char>(max_image, height, width, r.h, r.w);

                        unsigned char temp_dis = max_image_value - min_image_value;

                        if (temp_r > temp_dis)
                        {
                            set_pixel<unsigned char>(distance, height, width, r.h, r.w, temp_dis);
                            Q[temp_dis].push(r);
                        }
                    }

                    else if (state_value == 0)
                    {
                        unsigned char min_image_value = get_pixel<unsigned char>(min_image, height, width, p.h, p.w);
                        unsigned char max_image_value = get_pixel<unsigned char>(max_image, height, width, p.h, p.w);
                        set_pixel<unsigned char>(min_image, height, width, r.h, r.w, min_image_value);
                        set_pixel<unsigned char>(max_image, height, width, r.h, r.w, max_image_value);

                        unsigned char image_value = get_pixel<unsigned char>(img, height, width, r.h, r.w);

                        if (image_value < min_image_value)
                            set_pixel<unsigned char>(min_image, height, width, r.h, r.w, image_value);
                        if (image_value > max_image_value)
                            set_pixel<unsigned char>(max_image, height, width, r.h, r.w, image_value);

                        min_image_value = get_pixel<unsigned char>(min_image, height, width, r.h, r.w);
                        max_image_value = get_pixel<unsigned char>(max_image, height, width, r.h, r.w);

                        unsigned char temp_dis = max_image_value - min_image_value;

                        set_pixel<unsigned char>(distance, height, width, r.h, r.w, temp_dis);                   
                        Q[temp_dis].push(r);
                        set_pixel<int>(state, height, width, r.h, r.w, 1);
                    }
                    else
                        continue;

                }
            } 
        }
    }  
    delete state;
    delete min_image;
    delete max_image;
}