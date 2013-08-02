int writerealimage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize);

int writecompleximage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize);

int writeabsimage(TIFF* image, unsigned int width, unsigned int height, cuComplex *data, tsize_t scanlinesize);
