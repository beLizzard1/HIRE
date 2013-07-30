<<<<<<< HEAD
int writerealimage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize);

int writecompleximage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize);

int writeabsimage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize);
=======
int writeimage(TIFF* image, unsigned int width, unsigned int height, void *data, tsize_t scanlinesize);
>>>>>>> bc02b5056dc191a9d2a985d1c43dc343d9e1c91e
