#include <iostream>
#include <vector>
using namespace std;

struct Point2D
{
    // float distance;
    int w;
    int h;
};


void MBD_waterflow(const unsigned char * img, const unsigned char * seeds, unsigned char * distance, 
    int height, int width);