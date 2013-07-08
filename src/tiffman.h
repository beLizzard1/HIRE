int fileopen(char **filename, unsigned int *width, unsigned int *length, TIFF* image);
int filewrite(unsigned int width, unsigned int length,tmsize_t scanlinesize, float *data);
