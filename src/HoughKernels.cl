// OpenCL kernel functions.

// Horizontal pass of the 2-part Sobel operator process.
kernel void sobelOperatorHorizontal(global read_only int *inputImg, global write_only int *sobelOutX, global write_only int *sobelOutY, 
	const int width, const int height)
{

	// output image coordinates
	int x = get_global_id(0);
	int y = get_global_id(1);
	
	if(x >= width - 2 || y >= height)
		return;
	
	int index = y * (width - 2) + x;
	
	// input image coordinates
	int inIndex = y * width + x + 1;
	
	sobelOutX[index] = inputImg[inIndex - 1] - inputImg[inIndex + 1];
	sobelOutY[index] = inputImg[inIndex - 1] + 2 * inputImg[inIndex] + inputImg[inIndex + 1];
}

// Vertical pass of the 2-part Sobel operator process.
kernel void sobelOperatorVertical(global read_only int *sobelOutX, global read_only int *sobelOutY, global write_only int *sobelOut, 
	const int width, const int height)
{
	int newWidth = width - 2;
	
	// output image coordinates
	int x = get_global_id(0);
	int y = get_global_id(1);
	
	if(x >= width - 2 || y >= height - 2)
		return;
	
	int index = y * newWidth + x;
	
	// input image coordinates
	int inIndex = (y + 1) * newWidth + x;
	
	int gX = sobelOutX[inIndex - newWidth] + 2 * sobelOutX[inIndex] + sobelOutX[inIndex + newWidth];
	int gY = sobelOutY[inIndex - newWidth] - sobelOutY[inIndex + newWidth];
	
	sobelOut[index] = (int)native_sqrt((float)((gX * gX) + (gY * gY)));
}

// Hough circles.
kernel void houghCircles(global read_only int *houghIn, global read_write int *houghOut, const int threshold, const int width, 
	const int height, const int maxRadius)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int r = get_global_id(2) + 1;
	
	if(x >= width || y >= height || r > maxRadius)
		return;
	
	int index = width * y + x;
	
	if(houghIn[index] < threshold)
		return;
	
	int outX = x + maxRadius;
	int outY = y + maxRadius;
	int outWidth = width + 2 * maxRadius;
	int outSize = outWidth * (height + 2 * maxRadius);
	int outImage = r * outSize;
	
	for(int i = 0; i < 6 * r; i++) {
		float angle = i * 2 * 3.1416f / (6 * r);
		int a = outX - r * native_cos(angle);
		int b = outY - r * native_sin(angle);
		atomic_add(houghOut + outImage + b * outWidth + a, 1);
	}
}

kernel void findBiggest(global read_only int *biggestIn, global write_only int *biggestFound, const int numChunks, 
	const int chunkSize)
{
	int chunk = get_global_id(0);
	
	if(chunk >= numChunks)
		return;
	
	int2 biggestYet = (int2)(0, 0);
	
	int index = chunk * chunkSize;
	
	for(int i = 0; i < chunkSize; i++) {
		if(biggestIn[index + i] > biggestYet.x) {
			biggestYet.x = biggestIn[index + i];
			biggestYet.y = index + i;
		}
	}
	
	biggestFound[chunk * 2] = biggestYet.x;
	biggestFound[chunk * 2 + 1] = biggestYet.y;
}