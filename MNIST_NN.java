
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.Math;
import java.util.Random;
import javax.swing.*;

public class MNIST_NN {

	static final String BEST_WEIGHTS = "weights_9817_1783797742335epoch13.csv";

	double[][] layers;
	double[][] z_vals;

	double[][][] weights;
	double[][] biases;

	double[][][] weight_gradients; // amount to change each weight by
	double[][] bias_gradients; // same for biases

	double eta = 0.01; // value to multiply gradients by

	// one-hot vector representing which label is correct
	double[] y;

	String[] buf;


	static int[] sizes = {784, 256, 10};

	static final int NUM_TRAINING_EXAMPLES = 60000;
	static final int THROWAWAY_EPOCHS = 8;

	double[][] data;
	int[] labels;

	public static void main (String[] args) throws IOException {

		// initialize the net
		MNIST_NN net = new MNIST_NN();

		//train_shuffled(net, "mnist_train.csv", 30, true);

		net.load_weights(BEST_WEIGHTS);
		//test(net, "mnist_test.csv", true);

		SwingUtilities.invokeLater(() -> new DigitDemo(net));
		
	}

	public static void train(MNIST_NN net, String filename, int epochs, boolean save) throws IOException {

		net.eta = 0.05;
		
		for (int i = 0; i < epochs; i++) {
			double time = System.currentTimeMillis();
			System.out.println("Epoch: " + i);
			epoch(net, filename);
			net.eta *= 0.95;
			System.out.println("epoch time: " + (System.currentTimeMillis() - time) + "ms");
		}
		if (save) net.save_weights("../mnist_nn_logs/weights_" + System.currentTimeMillis() + ".csv");
	}


	public static void epoch(MNIST_NN net, String filename) throws IOException {

		try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
			// loop here
			int iters = 0;
			double window_sum = 0;
			int window_length = 20000;
			while (net.preprocess(br)) {
				window_sum += net.feed_forward();

				net.backprop();
				iters++;

				if (iters % window_length == 0) {
					System.out.println("Training steps: " + iters);
					System.out.println("Average cost since last report: " + window_sum / window_length);
					window_sum = 0;
				}
			}
			
		}
		
	}

	public static void train_shuffled(MNIST_NN net, String filename, int epochs, boolean save) throws IOException {

		net.eta = 0.05;

		Random rng = new Random();

		try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
			net.preprocess_array(br);
		}

		// for tracking hit rate
		double maxHR = 0;
		double currHR = 0;
		
		for (int i = 0; i < epochs; i++) {
			double time = System.currentTimeMillis();
			System.out.println("Epoch: " + i);
			epoch_shuffled(net, rng);
			net.eta *= 0.95;
			System.out.println("epoch time: " + (System.currentTimeMillis() - time) + "ms");

			currHR = test(net, "mnist_test.csv", false);
			if (currHR > maxHR) {
				maxHR = currHR;
				if (i > THROWAWAY_EPOCHS) {
					// dont save garbage weights from the first throwaway epochs
					net.save_weights(String.format("../mnist_nn_logs/weights_%04d_%depoch%d.csv",
						(int)(maxHR * 10000), System.currentTimeMillis(), i));
				}
				System.out.println("New best accuracy: " + maxHR);
				
			}
			
		}
		if (save) net.save_weights("../mnist_nn_logs/weights_" + System.currentTimeMillis() + "_epoch" + epochs +  ".csv");
	}


	public static void epoch_shuffled(MNIST_NN net, Random rng) {

		
		int[] order = new int[NUM_TRAINING_EXAMPLES];
		for (int i = 0; i < NUM_TRAINING_EXAMPLES; i++) {
			order[i] = i;
		}

		for (int i = order.length - 1; i > 0; i--) {
    		int j = rng.nextInt(i + 1);   // j in [0, i] — inclusive of i is the correct form
    		int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
		}
		// loop here
		int iters = 0;
		double window_sum = 0;
		int window_length = 20000;
		for (int i = 0; i < NUM_TRAINING_EXAMPLES; i++) {

			int idx = order[i];

			// setting example
			for (int j = 0; j < net.layers[0].length; j++) {
				net.layers[0][j] = net.data[idx][j];
			}
			for (int j = 0; j < net.y.length; j++) {
				net.y[j] = 0;
			}
			net.y[net.labels[idx]] = 1;

			window_sum += net.feed_forward();

			net.backprop();
			iters++;

			if (iters % window_length == 0) {
				System.out.println("Training steps: " + iters);
				System.out.println("Average cost since last report: " + window_sum / window_length);
				window_sum = 0;
			}
		}
			
		
		
	}


	public static double test(MNIST_NN net, String filename, boolean verbose) throws IOException {
		double hit_rate = 0;
		try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
			
			int tot_hits = 0;
			double max_activation = 0;
			int iters = 0;

			int[] hits = new int[sizes[sizes.length - 1]];
			int[] totals = new int[sizes[sizes.length - 1]];
			
			while (net.preprocess(br)) {
				net.feed_forward();

				int argmax = 0;
				double max = Double.MIN_VALUE;
				
				for (int i = 0; i < net.layers[sizes.length - 1].length; i++) {
					if (net.layers[sizes.length - 1][i] > max) {
						max = net.layers[sizes.length - 1][i];
						argmax = i;
					}
				}
				iters++;
				max_activation += max;
				if (net.y[argmax] == 1) { 
					tot_hits++;
					hits[argmax]++;
				}

				int yactual = 0;
				double ymax = Double.MIN_VALUE;
				for (int i = 0; i < net.y.length; i++) {
					if (net.y[i] > ymax) {ymax = net.y[i]; yactual = i;}
				}

				totals[yactual]++;
			}
			hit_rate = (double) tot_hits / iters;
			if (verbose) {
				System.out.println("Test results: \nExamples: " + iters);
				System.out.println("Average max activation: " + max_activation / iters);
				System.out.println("Hit rate: " + hit_rate);
				System.out.println("Stats per digit: ");
				for (int i = 0; i < sizes[sizes.length - 1]; i++) {
					System.out.println(i + ": " + (double) hits[i] / totals[i]);
				}
			}
		}
		return hit_rate;
	}

	// claude method
	public void save_weights(String filename) throws IOException {
		try (java.io.PrintWriter pw = new java.io.PrintWriter(new java.io.FileWriter(filename))) {

			// header: architecture
			for (int i = 0; i < sizes.length; i++) {
				pw.print(sizes[i] + (i < sizes.length - 1 ? "," : "\n"));
			}

			// weights: one line per neuron (row of weights[layer])
			for (int layer = 1; layer < sizes.length; layer++) {
				for (int j = 0; j < weights[layer].length; j++) {
					StringBuilder sb = new StringBuilder();
					for (int k = 0; k < weights[layer][j].length; k++) {
						if (k > 0) sb.append(',');
						sb.append(weights[layer][j][k]);
					}
					pw.println(sb);
				}
				// biases for this layer: one line
				StringBuilder sb = new StringBuilder();
				for (int j = 0; j < biases[layer].length; j++) {
					if (j > 0) sb.append(',');
					sb.append(biases[layer][j]);
				}
				pw.println(sb);
			}
		}
	}

	// claude method
	public void load_weights(String filename) throws IOException {
		try (BufferedReader br = new BufferedReader(new FileReader(filename))) {

			// header: verify architecture matches
			String[] header = br.readLine().split(",");
			if (header.length != sizes.length)
				throw new IOException("Architecture mismatch: file has " + header.length
						+ " layers, net has " + sizes.length);
			for (int i = 0; i < sizes.length; i++) {
				if (Integer.parseInt(header[i]) != sizes[i])
					throw new IOException("Architecture mismatch at layer " + i
							+ ": file " + header[i] + ", net " + sizes[i]);
			}

			// per layer: fanOut weight lines, then one bias line
			for (int layer = 1; layer < sizes.length; layer++) {
				for (int j = 0; j < weights[layer].length; j++) {
					String[] row = br.readLine().split(",");
					for (int k = 0; k < weights[layer][j].length; k++) {
						weights[layer][j][k] = Double.parseDouble(row[k]);
					}
				}
				String[] biasRow = br.readLine().split(",");
				for (int j = 0; j < biases[layer].length; j++) {
					biases[layer][j] = Double.parseDouble(biasRow[j]);
				}
			}
		}
	}

	

	public MNIST_NN() {
		layers = new double[sizes.length][];
		z_vals = new double[sizes.length][];

		

		for (int i = 0; i < layers.length; i++) {
			layers[i] = new double[sizes[i]];
			z_vals[i] = new double[sizes[i]];
		}

		init_weights();
		init_biases();

		
		y = new double[sizes[sizes.length-1]];

		data = new double[NUM_TRAINING_EXAMPLES][sizes[0]];
		labels = new int[NUM_TRAINING_EXAMPLES];


	}

	// initialize weights
	public void init_weights() {

		// weights[0] is null but this keeps the indexing scheme from 3b1b
		// weights[1] refers to weights from layer 0 to layer 1
		// same for biases

		weights = new double[sizes.length][][];
		weight_gradients = new double[sizes.length][][];

		Random rng = new Random();

		for (int endingLayer = 1; endingLayer < weights.length; endingLayer++) {
			int fanIn = sizes[endingLayer-1];
			int fanOut = sizes[endingLayer];

			weights[endingLayer] = new double[fanOut][fanIn];
			weight_gradients[endingLayer] = new double[fanOut][fanIn];

			// weight from a neuron in a layer of size fanIn:
			for (int i = 0; i < fanOut; i++) {
				for (int j = 0; j < fanIn; j++) {
					double weight = rng.nextGaussian() * Math.sqrt(2.0 / fanIn);
					weights[endingLayer][i][j] = weight;
				}
			}
		}
	}

	public void init_biases() {
		// biases[0] is null but this keeps the indexing scheme from 3b1b
		// biases[1] refers to biases from layer 0 to layer 1
		// biases[i][j] refers to the bias added to neuron j in layer i
		// 0-initialized
		biases = new double[sizes.length][];
		bias_gradients = new double[sizes.length][];

		for (int i = 1; i < sizes.length; i++) {
			biases[i] = new double[sizes[i]];
			bias_gradients[i] = new double[sizes[i]];
		}
	}
	

	public boolean preprocess(BufferedReader br) throws IOException {
		// assign activations to input layer (layers[0], a 784-element array)
		
		String line;
		if ((line = br.readLine()) != null) {
			buf = line.split(",");
			for (int i = 1; i < buf.length; i++) {
				layers[0][i-1] = Double.parseDouble(buf[i]) / 255.0;
			}

			int label = Integer.parseInt(buf[0]);
			for (int i = 0; i < y.length; i++) {
				y[i] = 0;
			}
			y[label] = 1;
			return true;
		}
		return false;
	}
	
	
	public void preprocess_array(BufferedReader br) throws IOException {
		// make a 60k x 784 array of all the examples to then shuffle in the training examples

		String line;

		int h = 0;


		while ((line = br.readLine()) != null) {
			buf = line.split(",");
			for (int i = 1; i < buf.length; i++) {
				data[h][i-1] = Double.parseDouble(buf[i]) / 255.0;
			}

			labels[h] = Integer.parseInt(buf[0]);
			h++;
			
			// todo: need to make sure that when i use this method the y-vector gets assigned appropriately at the start of each example
			// because the preprocess method assigns it 
		}
	}
	

	public double reLU(double n) {
		// leaky reLU
		return n >= 0 ? n : 0.01*n;
	}

	public double reLU_prime(double n) {
		return n >= 0 ? 1 : 0.01;
	}

	public double sigmoid(double n) {
		return 1/(1+Math.exp(-n));
	}

	public double sigmoid_prime(double n) {
		return sigmoid(n)*(1-sigmoid(n));
	}

	public double cost() {
		double[] output = layers[sizes.length - 1];
		double res = 0;
		for (int i = 0; i < output.length; i++) {
			
			double a = Math.max(1e-12, Math.min(1 - 1e-12, output[i]));
        	res += -(y[i]*Math.log(a) + (1 - y[i])*Math.log(1 - a));
		}
		return res;
		// average of all cost values is how badly the computer is doing
	}

	
	public void vec_mul(int layer) {
		
		// take weights[layer] and mat-mul with layers[layer-1], add biases[layer] to each element at the end

		// layer 1 sanity check: weights[1] is 16x784 and layers[0] is 784x1, so this works as planned.
		// layers[1] is 16 elements, each of which will get the big accumulated total plus the bias for that node.

		for (int i = 0; i < weights[layer].length; i++) {
			double total = 0;
			for (int j = 0; j < weights[layer][0].length; j++) {
				total += weights[layer][i][j] * layers[layer-1][j];
			}
			
			double z = total + biases[layer][i];

			// last layer has to use sigmoid
			layers[layer][i] = layer == sizes.length - 1 ? sigmoid(z) : reLU(z);
			z_vals[layer][i] = z;

		}

	}


	public double feed_forward() {
		// feed forward one training example
		// returns cost for this training example
		// parameter is the correct digit as labeled in the dataset
		
		// layer n = reLU(w*(layer n-1) + b), where w is a matrix of weights 
		// and row 0 holds the weights to be multiplied by the activation of each neuron in the previous layer for neuron 0 in the next layer, etc
		// and layer n-1 is the vector of activations in the previous layer
		// and b is a vector of biases
		for (int i = 1; i < sizes.length; i++) {
			vec_mul(i);
		}
		return cost();
	}

	public void backprop() {
		// compute the gradient of the cost wrt each node in the layer
		// adjust each weight and bias between these layers by the same fraction of that gradient
		// start with the output layer and repeat

		// (replace reLU with sigmoid in the final layer)

		// seems like the partials I need are, for each weight coming into this layer:
		// dC/dw_jk^(L), where w_jk^(L) is the weight from node k of the previous layer to node j of the current layer L
		// dC/dw_jk^(L) = a_k^(L-1) * reLU'(z_j^(L)) * 2(a_j^(L) - y_j), where y is a one-hot vector representing the correct label

		// AND in hidden layers i need to sum these partials for each node i.e. one addition to the sum for each weight that the node contributes??
		// so basically i need to hold onto what claude is calling the "error" from the previous calculation.
		// that error is sigmoid'(z_j^(L)) * 2(a_j^(L) - y_j) for the output layer. for upstream layers:

		// the general formula: δ_j = (activation′(z_j)) × (∂C/∂a_j)
		// this partial (∂C/∂a_j) is basically δ from the downstream layer. Is that correct?

		// and the partials for each bias are dC/db_j^L = 1 * reLU'(z_j^(L)) * 2(a_j^(L) - y_j).

		// put all the weight gradients in a matrix that's the same dimensions as weights[L]
		// and same for the column vector of bias gradients


		

		// will be non-null by the second iter
		double[] deltas_downstream = null;

		//backloop
		for (int layer = sizes.length-1; layer > 0; layer--) {

			double[] deltas_current = new double[sizes[layer]];

			if (layer == sizes.length-1) compute_deltas_output(deltas_current, layer);
			else compute_deltas_hidden(deltas_current, deltas_downstream, layer);


			// compute gradients
			// todo: if im going to do minibatching instead of one-by-one SGD
			// then ill need to zero the gradient arrays
			compute_gradients(deltas_current, layer);

			// switch current to downstream and reassign size of current
			deltas_downstream = deltas_current;
			
		}

		// second loop to adjust weights and biases 
		// (this is done to avoid updating weights before theyre used for computation of upstream layers' delta values)
		for (int layer = sizes.length-1; layer > 0; layer--) {
			update_weights_biases(layer);
		}

	}

	public void compute_deltas_output(double[] current, int layer) {
		for (int j = 0; j < current.length; j++) {
			current[j] = layers[layer][j]-y[j];
		}
	}

	public void compute_deltas_hidden(double[] current, double[] downstream, int layer) {

		for (int k = 0; k < current.length; k++) {
			
			double kSum = 0;
			for (int j = 0; j < downstream.length; j++) {
				// sum w_jk * delta j
				// weights [3] is 16x10 and k does 0 to 15, j goes 0 to 9
				kSum += weights[layer + 1][j][k] * downstream[j];
			}
			current[k] = reLU_prime(z_vals[layer][k]) * kSum;

		}

	}

	public void compute_gradients(double[] current_deltas, int layer) {
		// again we're computing the gradients which will be multiplied by a small factor
		// and subtracted from each weights[layer]

		for (int j = 0; j < weight_gradients[layer].length; j++) {
			for (int k = 0; k < weight_gradients[layer][j].length; k++) {
				weight_gradients[layer][j][k] = current_deltas[j] * layers[layer-1][k];
				
			}
			bias_gradients[layer][j] = current_deltas[j];
		}
	}

	public void update_weights_biases(int layer) {

		for (int j = 0; j < weights[layer].length; j++) {
			for (int k = 0; k < weights[layer][j].length; k++) {
				weights[layer][j][k] -= eta * weight_gradients[layer][j][k];
				
			}
			biases[layer][j] -= eta * bias_gradients[layer][j];
		}
	}

}

