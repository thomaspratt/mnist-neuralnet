import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * Draw a digit, watch the net classify it.
 *
 * Usage:
 *   MNIST_NN net = new MNIST_NN();
 *   net.load_weights("weights.csv");
 *   SwingUtilities.invokeLater(() -> new DigitDemo(net));
 *
 * Or run this class's main() directly (edit WEIGHTS_FILE below).
 *
 * The important part is preprocess(): it reproduces MNIST's own pipeline
 * (crop to ink -> scale to fit 20x20 -> paste into 28x28 with the center
 * of mass at the middle -> grayscale from the downscale). Skipping this
 * and just shrinking the canvas is why most homemade MNIST demos fail.
 */
public class DigitDemo extends JFrame {

	static final String WEIGHTS_FILE = "weights.csv"; // edit me

	static final int CANVAS = 280;      // drawing area, 10x the MNIST frame
	static final int BRUSH = 18;        // stroke width in canvas pixels

	final MNIST_NN net;

	BufferedImage canvas;               // what the user draws on
	BufferedImage netInput;             // the 28x28 the net actually sees
	double[] outputs = null;            // last output-layer activations
	int argmax = -1;

	Point lastDrag = null;

	JPanel drawPanel, previewPanel, barsPanel;

	public DigitDemo(MNIST_NN net) {
		super("MNIST demo — draw a digit");
		this.net = net;

		canvas = new BufferedImage(CANVAS, CANVAS, BufferedImage.TYPE_BYTE_GRAY);
		clearCanvas();

		// ---------- drawing panel ----------
		drawPanel = new JPanel() {
			@Override protected void paintComponent(Graphics g) {
				super.paintComponent(g);
				g.drawImage(canvas, 0, 0, null);
				g.setColor(Color.DARK_GRAY);
				g.drawRect(0, 0, CANVAS - 1, CANVAS - 1);
			}
		};
		drawPanel.setPreferredSize(new Dimension(CANVAS, CANVAS));

		MouseAdapter drawing = new MouseAdapter() {
			@Override public void mousePressed(MouseEvent e) {
				lastDrag = e.getPoint();
				stroke(lastDrag, lastDrag);   // dot on click
			}
			@Override public void mouseDragged(MouseEvent e) {
				Point p = e.getPoint();
				stroke(lastDrag, p);
				lastDrag = p;
			}
			@Override public void mouseReleased(MouseEvent e) {
				lastDrag = null;
				classify();                   // classify on pen-up
			}
		};
		drawPanel.addMouseListener(drawing);
		drawPanel.addMouseMotionListener(drawing);

		// ---------- preview: the 28x28 the net sees, blown up 4x ----------
		previewPanel = new JPanel() {
			@Override protected void paintComponent(Graphics g) {
				super.paintComponent(g);
				g.setColor(Color.BLACK);
				g.fillRect(0, 0, 112, 112);
				if (netInput != null) {
					// nearest-neighbor upscale so the pixels stay blocky (that's the point)
					g.drawImage(netInput, 0, 0, 112, 112, null);
				}
				g.setColor(Color.DARK_GRAY);
				g.drawRect(0, 0, 111, 111);
			}
		};
		previewPanel.setPreferredSize(new Dimension(112, 130));
		previewPanel.setBorder(BorderFactory.createTitledBorder("net's view"));

		// ---------- bars: all ten output activations ----------
		barsPanel = new JPanel() {
			@Override protected void paintComponent(Graphics g) {
				super.paintComponent(g);
				int w = getWidth(), h = getHeight();
				int barW = w / 10;
				g.setFont(getFont().deriveFont(Font.BOLD, 14f));
				for (int d = 0; d < 10; d++) {
					double a = outputs == null ? 0 : outputs[d];
					int barH = (int) (a * (h - 30));
					int x = d * barW;
					g.setColor(d == argmax ? new Color(220, 60, 60) : new Color(90, 90, 200));
					g.fillRect(x + 4, h - 22 - barH, barW - 8, barH);
					g.setColor(Color.BLACK);
					g.drawString(Integer.toString(d), x + barW / 2 - 4, h - 6);
				}
			}
		};
		barsPanel.setPreferredSize(new Dimension(CANVAS, 140));
		barsPanel.setBorder(BorderFactory.createTitledBorder("output activations"));

		// ---------- buttons ----------
		JButton clear = new JButton("Clear");
		clear.addActionListener(e -> {
			clearCanvas();
			netInput = null;
			outputs = null;
			argmax = -1;
			repaintAll();
		});

		JPanel side = new JPanel();
		side.setLayout(new BoxLayout(side, BoxLayout.Y_AXIS));
		side.add(previewPanel);
		side.add(Box.createVerticalStrut(10));
		side.add(clear);

		JPanel top = new JPanel(new BorderLayout(10, 10));
		top.add(drawPanel, BorderLayout.CENTER);
		top.add(side, BorderLayout.EAST);

		setLayout(new BorderLayout(10, 10));
		add(top, BorderLayout.CENTER);
		add(barsPanel, BorderLayout.SOUTH);

		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pack();
		setLocationRelativeTo(null);
		setVisible(true);
	}

	void clearCanvas() {
		Graphics2D g = canvas.createGraphics();
		g.setColor(Color.BLACK);              // MNIST: black background,
		g.fillRect(0, 0, CANVAS, CANVAS);     // white ink
		g.dispose();
	}

	void stroke(Point a, Point b) {
		Graphics2D g = canvas.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.setColor(Color.WHITE);
		g.setStroke(new BasicStroke(BRUSH, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
		g.drawLine(a.x, a.y, b.x, b.y);
		g.dispose();
		drawPanel.repaint();
	}

	/**
	 * Canvas -> 784-array, reproducing MNIST's documented preprocessing:
	 * bounding box of ink -> scale to fit 20x20 (aspect preserved) ->
	 * place in 28x28 with center of mass at (14,14) -> normalize to [0,1].
	 */
	double[] preprocess() {
		// 1. bounding box of ink on the big canvas
		int minX = CANVAS, minY = CANVAS, maxX = -1, maxY = -1;
		for (int yPix = 0; yPix < CANVAS; yPix++) {
			for (int x = 0; x < CANVAS; x++) {
				if ((canvas.getRGB(x, yPix) & 0xFF) > 0) {
					if (x < minX) minX = x;
					if (x > maxX) maxX = x;
					if (yPix < minY) minY = yPix;
					if (yPix > maxY) maxY = yPix;
				}
			}
		}
		if (maxX < 0) return null; // blank canvas

		int bw = maxX - minX + 1, bh = maxY - minY + 1;

		// 2. scale the crop so its longer side becomes 20 px
		double scale = 20.0 / Math.max(bw, bh);
		int sw = Math.max(1, (int) Math.round(bw * scale));
		int sh = Math.max(1, (int) Math.round(bh * scale));

		BufferedImage scaled = new BufferedImage(sw, sh, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g = scaled.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
				RenderingHints.VALUE_INTERPOLATION_BILINEAR); // soft edges = MNIST-like grayscale
		g.drawImage(canvas.getSubimage(minX, minY, bw, bh), 0, 0, sw, sh, null);
		g.dispose();

		// 3. center of mass of the scaled crop
		double mass = 0, mx = 0, my = 0;
		for (int yPix = 0; yPix < sh; yPix++) {
			for (int x = 0; x < sw; x++) {
				int v = scaled.getRGB(x, yPix) & 0xFF;
				mass += v;
				mx += v * x;
				my += v * yPix;
			}
		}
		mx /= mass;
		my /= mass;

		// 4. paste into 28x28 so the COM lands at (14,14); clamp so it fits
		int offX = (int) Math.round(14 - mx);
		int offY = (int) Math.round(14 - my);
		offX = Math.max(0, Math.min(28 - sw, offX));
		offY = Math.max(0, Math.min(28 - sh, offY));

		netInput = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g2 = netInput.createGraphics();
		g2.setColor(Color.BLACK);
		g2.fillRect(0, 0, 28, 28);
		g2.drawImage(scaled, offX, offY, null);
		g2.dispose();

		// 5. flatten row-major, normalize to [0,1] — same as training
		double[] out = new double[784];
		for (int yPix = 0; yPix < 28; yPix++) {
			for (int x = 0; x < 28; x++) {
				out[yPix * 28 + x] = (netInput.getRGB(x, yPix) & 0xFF) / 255.0;
			}
		}
		return out;
	}

	void classify() {
		double[] input = preprocess();
		if (input == null) return;

		System.arraycopy(input, 0, net.layers[0], 0, 784);
		net.y = new double[MNIST_NN.sizes[MNIST_NN.sizes.length - 1]]; // dummy: feed_forward's cost() reads y
		net.feed_forward();

		outputs = net.layers[MNIST_NN.sizes.length - 1].clone();
		argmax = 0;
		for (int i = 1; i < outputs.length; i++)
			if (outputs[i] > outputs[argmax]) argmax = i;

		repaintAll();
	}

	void repaintAll() {
		drawPanel.repaint();
		previewPanel.repaint();
		barsPanel.repaint();
	}

	public static void main(String[] args) throws IOException {
		MNIST_NN net = new MNIST_NN();
		net.load_weights(args.length > 0 ? args[0] : WEIGHTS_FILE);
		SwingUtilities.invokeLater(() -> new DigitDemo(net));
	}
}
