import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Paint;
import java.awt.Shape;
import java.awt.geom.Ellipse2D;
import java.awt.image.BufferedImage;
import java.io.File;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;

public class Main {

	public static void main(String[] args) throws Exception {
		BufferedImage input = ImageIO.read(new File(args[0]));
		
		HoughCircles hough = new HoughCircles(1, 0); // CL platform, CL device
//		for(int i = 0; i < 100; i++)
//			hough.process(input);
//		long time = System.currentTimeMillis();
//		for(int i = 0; i < 1000; i++)
//			hough.process(input);
//		System.out.println((System.currentTimeMillis() - time) / 1000);
		
		int scale = Integer.parseInt(args[1]);
		
		BufferedImage small = new BufferedImage(input.getWidth() / scale, input.getHeight() / scale, BufferedImage.TYPE_BYTE_GRAY);
		Graphics g = small.createGraphics();
		g.drawImage(input, 0, 0, input.getWidth() / scale, input.getHeight() / scale, null);
		g.dispose();
		show(small);
		hough.process(small);
		
//		for(int i = 0; i < 10; i++)
//			hough.process(small);
//		long time = System.currentTimeMillis();
//		for(int i = 0; i < 10; i++)
//			hough.process(small);
//		System.out.println((System.currentTimeMillis() - time) / 10);
		
		show(hough.getSobel());
		
		Graphics2D graphic = input.createGraphics();
		graphic.setColor(Color.green);
		
		int n = 3;
        graphic.setStroke(new BasicStroke(2.0f));
		graphic.draw(new Ellipse2D.Double((hough.getCentre().getX() - hough.getRadius() + n) * scale, (hough.getCentre().getY() - hough.getRadius() + n) * scale, 
				hough.getRadius() * 2 * scale, hough.getRadius() * 2 * scale));
		
		show(input);
		
		hough.close();
	}
	
	public static void show(final BufferedImage image) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame frame = new JFrame();
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.add(new JLabel(new ImageIcon(image)));
                frame.pack();
                frame.setLocation(0, 0);
                frame.setVisible(true);
            }
        });
    }

}
