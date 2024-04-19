import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import components.simplereader.SimpleReader;
import components.simplereader.SimpleReader1L;

/**
 * {@code CNNLayers} represented as a {@code double[][]} with implementations of
 * primary methods.
 *
 * @correspondence <pre>
 * this.is_valid = [$this.weights is valid] and
 *  this.weights.length = $this.inLength
 *  this.weights[0].length = $this.outLength and
 *  if $this.next != null then
 *    this = $this.next.previous and
 *  if $this.previous != null then
 *    this = $this.previous.next
 * </pre>
 * @convention {@code
 * [$this.weights is not null when the CNN is propagating and
 *  the number of rows in the weight matrix is equal to the layer's input length
 *  and the number of columns in the weight matrix is equal to the layer's
 *  output length. If $this.next exists, this is equal to $this.previous of
 *  $this.next. If $this.previous exists, this is equal to $this.next of
 *  $this.previous]
 * }
 *
 * @author Sepehr Hooshiari
 *
 */
public class CNNLayers1 extends CNNLayersSecondary {

    /**
     * Keeps track of next layer.
     */
    private CNNLayers next;

    /**
     * Keeps track of previous layer.
     */
    private CNNLayers previous;

    /**
     * Input data corresponding to an image.
     */
    private double[][] data;

    /**
     * Label of the input data which indicates what number the image represents.
     */
    private int label;

    /**
     * The number of inputs to this layer.
     */
    private int inLength;

    /**
     * The array of inputs to this layer.
     */
    private double[] inputs;

    /**
     * The number of outputs from this layer.
     */
    private int outLength;

    /**
     * The array of outputs from this layer.
     */
    private double[] outputs;

    /**
     * Seed to generate matrix of random weights.
     */
    private long seed;

    /**
     * Rate at which the CNN should learn.
     */
    private double learningRate;

    /**
     * Representation of {@code this}.
     */
    private double[][] weights;

    /**
     * The number of layers in the CNN.
     */
    private List<CNNLayers> layers;

    /**
     * Creator of initial representation.
     */
    private void createNewRep() {
        this.inLength = 0;
        this.outLength = 0;
        this.seed = 0;
        this.learningRate = 0;
        this.weights = new double[0][0];
        this.layers = new ArrayList<>();
        this.setWeights();
        this.setLayers();
    }

    /**
     * Private methods. -------------------------------------------------------
     */

    /**
     * Sets the initial weights with a Gaussian distribution around 0.
     *
     * @ensures <pre> this.weights.length = this.inLength and
     * this.weights[0].length = this.outLength </pre>
     */
    private void setWeights() {
        Random rand = new Random(this.seed);
        for (int i = 0; i < this.inLength; i++) {
            for (int j = 0; j < this.outLength; j++) {
                // use nextGaussian so that random weights are distributed
                // close to 0
                this.weights[i][j] = rand.nextGaussian();
            }
        }
    }

    /**
     * Uses the Sigmoid function on {@code sum} to activate {@code sum}.
     *
     * @param sum
     *            the weighted sum of weights and their weights
     *
     * @return the activated weighted summation of {@code sum}
     */
    private double activation(double sum) {
        return 1 / (1 + Math.pow(Math.E, -sum));
    }

    /**
     * Sets each layer of the CNN with links to previous and next layers.
     */
    private void setLayers() {
        if (this.layers.size() > 1) {
            for (int i = 0; i < this.layers.size(); i++) {
                if (i == 0) {
                    this.layers.get(i).setNext(this.layers.get(i + 1));
                } else if (i == this.layers.size() - 1) {
                    this.layers.get(i).setPrevious(this.layers.get(i - 1));
                } else {
                    this.layers.get(i).setNext(this.layers.get(i + 1));
                    this.layers.get(i).setPrevious(this.layers.get(i - 1));
                }
            }
        }
    }

    /**
     * Constructors. -----------------------------------------------------------
     */

    /**
     * No-argument constructor for {@code this}.
     */
    public CNNLayers1() {
        this.createNewRep();
    }

    /**
     * Constructor for {@code this}.
     *
     * @param inLength
     *            the length of inputs to the layer
     * @param outLength
     *            the length of the layer's outputs
     * @param seed
     *            seed to generate initially random weights
     * @param learningRate
     *            the rate at which the CNN learns
     */
    public CNNLayers1(int inLength, int outLength, long seed,
            double learningRate) {
        this.inLength = inLength;
        this.outLength = outLength;
        this.seed = seed;
        this.learningRate = learningRate;
        this.weights = new double[inLength][outLength];
        this.layers = new ArrayList<>();
        this.setWeights();
        this.setLayers();
    }

    /**
     * Standard methods. -------------------------------------------------------
     */

    @Override
    public final CNNLayers newInstance() {
        try {
            return this.getClass().getConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(
                    "Cannot construct object of type " + this.getClass());
        }
    }

    @Override
    public final void clear() {
        this.createNewRep();
    }

    @Override
    public final void transferFrom(CNNLayers source) {
        assert source != null : "Violation of: source is not null";
        assert source != this : "Violation of: source is not this";
        assert source instanceof CNNLayers1 : ""
                + "Violation of: source is of dynamic type SimpleReader1L";
        /*
         * This cast cannot fail since the assert above would have stopped
         * execution in that case.
         */
        CNNLayers1 localSource = (CNNLayers1) source;
        this.weights = localSource.weights;
        this.inLength = localSource.inLength;
        this.outLength = localSource.outLength;
        this.learningRate = localSource.learningRate;
        this.seed = localSource.seed;
        localSource.createNewRep();
    }

    /**
     * Kernel methods. -----------------------------------------------
     */

    @Override
    public final List<CNNLayers> getLayers() {
        return this.layers;
    }

    @Override
    public final void copyFrom(CNNLayers source) {
        assert source != null : "Violation of: source is not null";
        assert source != this : "Violation of: source is not this";
        assert source instanceof CNNLayers1 : ""
                + "Violation of: source is of dynamic type SimpleReader1L";
        /*
         * This cast cannot fail since the assert above would have stopped
         * execution in that case.
         */
        CNNLayers1 localSource = (CNNLayers1) source;
        this.weights = localSource.weights;
        this.inLength = localSource.inLength;
        this.outLength = localSource.outLength;
        this.learningRate = localSource.learningRate;
        this.seed = localSource.seed;
    }

    @Override
    public final double[][] multiplyMatrix(double[][] x, double scalar) {
        double[][] output = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                output[i][j] = x[i][j] * scalar;
            }
        }
        return output;
    }

    @Override
    public final List<double[][]> analyze(String s) {
        List<double[][]> images = new ArrayList<>();
        final int rows = 28;
        final int columns = 28;
        try (SimpleReader in = new SimpleReader1L(s)) {
            String line = "";
            while (!in.atEOS()) {
                line = in.nextLine();
                String[] items = line.split(",");
                this.data = new double[rows][columns];
                this.label = Integer.parseInt(items[0]);
                int i = 1;
                for (int j = 0; j < rows; j++) {
                    for (int k = 0; k < columns; k++) {
                        this.data[j][k] = Double.parseDouble(items[i]);
                        i++;
                    }
                }
                images.add(this.data);
            }
        } catch (Exception e) {
            throw new AssertionError("Violation of: can read from file");
        }
        return images;
    }

    @Override
    public final void addLayer(CNNLayers layer) {
        this.layers.add(layer);
        this.setLayers();
    }

    @Override
    public final CNNLayers getNext() {
        CNNLayers nextLayer = new CNNLayers1();
        if (this.next != null) {
            nextLayer = this.next;
        }
        return nextLayer;
    }

    @Override
    public final void setNext(CNNLayers nextLayer) {
        this.next = new CNNLayers1();
        this.next.copyFrom(nextLayer);
    }

    @Override
    public final CNNLayers getPrevious() {
        CNNLayers previousLayer = new CNNLayers1();
        if (this.previous != null) {
            previousLayer = this.previous;
        }
        return previousLayer;
    }

    @Override
    public final void setPrevious(CNNLayers previousLayer) {
        this.previous = new CNNLayers1();
        this.previous.copyFrom(previousLayer);
    }

    @Override
    public final double[][] getData() {
        this.analyze("data/mnist_test.csv");
        return this.data;
    }

    @Override
    public final int getLabel() {
        this.analyze("data/mnist_test.csv");
        return this.label;
    }

    @Override
    public final int inputLength() {
        return this.inLength;
    }

    @Override
    public final int outputLength() {
        return this.outLength;
    }

    @Override
    public final double[] getInputs() {
        double[] in = new double[this.inLength];
        if (this.inputs != null) {
            in = this.inputs;
        }
        return in;
    }

    @Override
    public final double[] getOutputs() {
        double[] out = new double[this.outLength];
        if (this.outputs != null) {
            out = this.outputs;
        }
        return out;
    }

    @Override
    public final double[][] getWeights() {
        return this.weights;
    }

    @Override
    public final void setWeightsIn(double[][] weights) {
        this.weights = weights;
    }

    @Override
    public final double getLearnRate() {
        return this.learningRate;
    }

    @Override
    public final boolean isValid() {
        return this.weights != null;
    }

    @Override
    public final double[] toArray(List<double[][]> input) {
        int length = input.size();
        int rows = input.get(0).length;
        int columns = input.get(0)[0].length;

        double[] arr = new double[length * rows * columns];

        int i = 0;
        for (int j = 0; j < length; j++) {
            for (int k = 0; k < rows; k++) {
                for (int l = 0; l < columns; l++) {
                    arr[i] = input.get(j)[k][l];
                    i++;
                }
            }
        }
        return arr;
    }

    @Override
    public final List<double[][]> toMatrix(double[] input, int length, int rows,
            int columns) {
        assert input.length == rows * columns
                * length : "Violation of : length out of bounds";
        List<double[][]> matrices = new ArrayList<>();
        int i = 0;
        for (int j = 0; j < length; j++) {
            double[][] m = new double[rows][columns];
            for (int k = 0; k < rows; k++) {
                for (int l = 0; l < columns; l++) {
                    m[k][l] = input[i];
                    i++;
                }
            }
            matrices.add(m);
        }
        return matrices;
    }

    @Override
    public final double[] forwardPass(double[] input) {
        this.inputs = input;
        double[] out1 = new double[this.outLength];
        double[] out2 = new double[this.outLength];

        for (int i = 0; i < this.inLength; i++) {
            for (int j = 0; j < this.outLength; j++) {
                out1[j] += input[i] * this.weights[i][j];
            }
        }
        this.outputs = out1;

        for (int i = 0; i < this.inLength; i++) {
            for (int j = 0; j < this.outLength; j++) {
                out2[j] = this.activation(out1[j]);
            }
        }
        return out2;
    }

    @Override
    public final double sigmoidPrime(double input) {
        final double leak = 0.01;
        double out = this.activation(input) * (1 - this.activation(input));
        if (out == 0) {
            out = leak;
        }
        return out;
    }

}
