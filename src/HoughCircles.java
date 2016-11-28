import java.awt.Graphics;
import java.awt.Image;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.awt.image.FilteredImageSource;
import java.awt.image.ImageFilter;
import java.awt.image.ImageProducer;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;

import javax.swing.GrayFilter;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLCommandQueue.Mode;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;

public class HoughCircles implements AutoCloseable
{

	private CLPlatform platform;
	private CLContext context;
	private CLProgram program;
	private CLCommandQueue queue;
	private CLKernel sobelOperatorHorizontal;
	private CLKernel sobelOperatorVertical;
	private CLKernel houghCircles;
	private CLKernel findBiggest;
	private int localWorkSize;
	private BufferedImage sobel;
	
	private int radius;
	private Point centre;
	
	
	public HoughCircles(int platformNum, int deviceNum) throws IOException
	{
		platform = CLPlatform.listCLPlatforms()[platformNum];
		context = CLContext.create(platform.listCLDevices()[deviceNum]);


		program = context.createProgram(getStreamFor("HoughKernels.cl"));
		program.build();

		queue = context.getDevices()[0].createCommandQueue();

		sobelOperatorHorizontal = program.createCLKernel("sobelOperatorHorizontal");
		sobelOperatorVertical = program.createCLKernel("sobelOperatorVertical");
		houghCircles = program.createCLKernel("houghCircles");
		findBiggest = program.createCLKernel("findBiggest");
		
		localWorkSize = queue.getDevice().getMaxWorkGroupSize();
		
		centre = new Point();
	}
	
	public void process(BufferedImage input)
	{
		BufferedImage greyscale = new BufferedImage(input.getWidth(), input.getHeight(), BufferedImage.TYPE_BYTE_GRAY);  
		Graphics g = greyscale.getGraphics();  
		g.drawImage(input, 0, 0, null);

		
		int pixels[] = greyscale.getRaster().getPixels(0, 0, greyscale.getWidth(), greyscale.getHeight(), (int[])null);
		
		IntBuffer inputIntBuffer = Buffers.newDirectIntBuffer(pixels);
		IntBuffer sobelOutXIntBuffer = Buffers.newDirectIntBuffer((input.getWidth() - 2) * input.getHeight());
		IntBuffer sobelOutYIntBuffer = Buffers.newDirectIntBuffer((input.getWidth() - 2) * input.getHeight());
		IntBuffer sobelInXIntBuffer = Buffers.newDirectIntBuffer((input.getWidth() - 2) * input.getHeight());
		IntBuffer sobelInYIntBuffer = Buffers.newDirectIntBuffer((input.getWidth() - 2) * input.getHeight());

		IntBuffer sobelOutIntBuffer = Buffers.newDirectIntBuffer((input.getWidth() - 2) * (input.getHeight() - 2));
		
		int maxRadius = input.getHeight();
		
		IntBuffer houghInIntBuffer = Buffers.newDirectIntBuffer((input.getWidth() - 2) * (input.getHeight() - 2));
		IntBuffer houghOutIntBuffer = Buffers.newDirectIntBuffer(((input.getWidth() - 2) + 2 * maxRadius) * ((input.getHeight() - 2) + 2 * maxRadius)
				* maxRadius);
		
		IntBuffer biggestInIntBuffer = Buffers.newDirectIntBuffer(((input.getWidth() - 2) + 2 * maxRadius) * ((input.getHeight() - 2) + 2 * maxRadius)
				* maxRadius);
		int chunks = 2048;
		IntBuffer biggestFoundIntBuffer = Buffers.newDirectIntBuffer(chunks);

		CLBuffer<IntBuffer> inputBuffer = context.createBuffer(inputIntBuffer, CLBuffer.Mem.READ_ONLY);
		CLBuffer<IntBuffer> sobelOutXBuffer = context.createBuffer(sobelOutXIntBuffer, CLBuffer.Mem.WRITE_ONLY);
		CLBuffer<IntBuffer> sobelOutYBuffer = context.createBuffer(sobelOutYIntBuffer, CLBuffer.Mem.WRITE_ONLY);
		CLBuffer<IntBuffer> sobelInXBuffer = context.createBuffer(sobelInXIntBuffer, CLBuffer.Mem.READ_ONLY);
		CLBuffer<IntBuffer> sobelInYBuffer = context.createBuffer(sobelInYIntBuffer, CLBuffer.Mem.READ_ONLY);

		CLBuffer<IntBuffer> sobelOutBuffer = context.createBuffer(sobelOutIntBuffer, CLBuffer.Mem.WRITE_ONLY);
		
		CLBuffer<IntBuffer> houghInBuffer = context.createBuffer(houghInIntBuffer, CLBuffer.Mem.READ_ONLY);
		CLBuffer<IntBuffer> houghOutBuffer = context.createBuffer(houghOutIntBuffer, CLBuffer.Mem.READ_WRITE);
		
		CLBuffer<IntBuffer> biggestInBuffer = context.createBuffer(biggestInIntBuffer, CLBuffer.Mem.READ_ONLY);
		CLBuffer<IntBuffer> biggestFoundBuffer = context.createBuffer(biggestFoundIntBuffer, CLBuffer.Mem.WRITE_ONLY);
		
		
		
		int[] sobelHGlobalWorkSize = new int[2];
		
		sobelHGlobalWorkSize[0] = (input.getWidth() - 2 + 63) / 64 * 64;
		sobelHGlobalWorkSize[1] = (input.getHeight() + 63) / 64 * 64;
		
		int[] sobelVGlobalWorkSize = new int[2];
		
		sobelVGlobalWorkSize[0] = (input.getWidth() - 2 + 63) / 64 * 64;
		sobelVGlobalWorkSize[1] = (input.getHeight() - 2 + 63) / 64 * 64;
		
		sobelOperatorHorizontal.putArg(inputBuffer)
								.putArg(sobelOutXBuffer)
								.putArg(sobelOutYBuffer)
								.putArg(input.getWidth())
								.putArg(input.getHeight())
								.rewind();
		sobelOperatorVertical.putArg(sobelInXBuffer)
								.putArg(sobelInYBuffer)
								.putArg(sobelOutBuffer)
								.putArg(input.getWidth())
								.putArg(input.getHeight())
								.rewind();
		houghCircles.putArg(houghInBuffer)
								.putArg(houghOutBuffer)
								.putArg(200)
								.putArg(input.getWidth() - 2)
								.putArg(input.getHeight() - 2)
								.putArg(maxRadius)
								.rewind();
		findBiggest.putArg(biggestInBuffer)
								.putArg(biggestFoundBuffer)
								.putArg(chunks)
								.putArg(houghOutBuffer.getBuffer().capacity() / chunks)
								.rewind();
		
		
		queue.putWriteBuffer(inputBuffer, false);
//		queue.putWriteBuffer(sobelOutXBuffer, false);
//		queue.putWriteBuffer(sobelOutYBuffer, false);
//		queue.putWriteBuffer(sobelInXBuffer, false);
//		queue.putWriteBuffer(sobelInYBuffer, false);
//		queue.putWriteBuffer(sobelOutBuffer, false);
		queue.put2DRangeKernel(sobelOperatorHorizontal, 0, 0, sobelHGlobalWorkSize[0], sobelHGlobalWorkSize[1], 
				0, 0);
		queue.putCopyBuffer(sobelOutXBuffer, sobelInXBuffer);
		queue.putCopyBuffer(sobelOutYBuffer, sobelInYBuffer);
		queue.put2DRangeKernel(sobelOperatorVertical, 0, 0, sobelVGlobalWorkSize[0], sobelVGlobalWorkSize[1], 
				0, 0);
		queue.putReadBuffer(sobelOutBuffer, false);
		queue.putCopyBuffer(sobelOutBuffer, houghInBuffer);
//		queue.put3DRangeKernel(houghCircles, 0, 0, 0, (input.getWidth() - 2 + 63) / 64 * 64, (input.getHeight() + 63) / 64 * 64, (maxRadius + 63) / 64 * 64, 0, 0, 0);
		
		for(int i = 0; i < (maxRadius) / 64 * 64; i += 64) {
			queue.put3DRangeKernel(houghCircles, 0, 0, i, (input.getWidth() - 2 + 63) / 64 * 64, (input.getHeight() + 63) / 64 * 64, 64, 0, 0, 0);
		}
		
		queue.putCopyBuffer(houghOutBuffer, biggestInBuffer);
		queue.put1DRangeKernel(findBiggest, 0, houghOutBuffer.getBuffer().capacity() / chunks, 0);
		queue.putReadBuffer(biggestFoundBuffer, true);
		
		sobel = new BufferedImage(input.getWidth() - 2, input.getHeight() - 2, BufferedImage.TYPE_BYTE_GRAY);
		int[] sobelPix = new int[sobelOutBuffer.getBuffer().capacity()];
		sobelOutBuffer.getBuffer().get(sobelPix).rewind();
		for(int i = 0; i < sobelPix.length; i++) {
			sobelPix[i] /= 6;
		}
		sobel.getRaster().setPixels(0, 0, input.getWidth() - 2, input.getHeight() - 2, sobelPix);
		
		int[] biggestPoints = new int[biggestFoundBuffer.getBuffer().capacity()];
		biggestFoundBuffer.getBuffer().get(biggestPoints).rewind();
		
		int highest = 0;
		int index = 0;
		for(int i = 0; i < biggestFoundBuffer.getBuffer().capacity() / 2; i++) {
			if(biggestPoints[2 * i] > highest) {
				highest = biggestPoints[2 * i];
				index = biggestPoints[2 * i + 1];
			}
		}
		
		radius = index / (((input.getWidth() - 2) + 2 * maxRadius) * ((input.getHeight() - 2) + 2 * maxRadius));
		int remainder = index % (((input.getWidth() - 2) + 2 * maxRadius) * ((input.getHeight() - 2) + 2 * maxRadius));
		centre.y = remainder / ((input.getWidth() - 2) + 2 * maxRadius) - 1;
		remainder %= ((input.getWidth() - 2) + 2 * maxRadius);
		centre.x = remainder - 1;
		
		centre.x -= maxRadius;
		centre.y -= maxRadius;
		
		inputBuffer.release();
		sobelOutXBuffer.release();
		sobelOutYBuffer.release();
		sobelInXBuffer.release();
		sobelInYBuffer.release();

		sobelOutBuffer.release();
		
		houghInBuffer.release();
		houghOutBuffer.release();
		biggestInBuffer.release();
		biggestFoundBuffer.release();
	}
	
	public BufferedImage getSobel()
	{
		return sobel;
	}
	
	public int getRadius()
	{
		return radius;
	}
	
	public Point getCentre()
	{
		return centre;
	}
	
	private static InputStream getStreamFor(String filename) {
        return HoughCircles.class.getResourceAsStream(filename);
    }

	@Override
	public void close() throws Exception {
		context.release();
	}
}
