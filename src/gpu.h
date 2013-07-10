int gpusqrt(float *data, unsigned int width, unsigned int height);
int gpudistance(float *distance, unsigned int width, unsigned int height);
int gpurefwavecalc(cuComplex *referencewave, float *data, float *distancegrid, float k, unsigned int width, unsigned int height);
int subimagedivref(cuComplex *reducedhologram, float * subimage, cuComplex *refwave, unsigned int width, unsigned int height);
