__kernel void vaad(__global float* a, __global float* b, __global float * c, const unsigned int size){
	int i = get_global_id(0);
	if (i<size)
		c[i] = a[i] + b[i];
}
