import static java.util.Collections.shuffle;

import java.util.ArrayList;
import java.util.List;

import components.simplereader.SimpleReader;
import components.simplereader.SimpleReader1L;
import components.simplewriter.SimpleWriter;
import components.simplewriter.SimpleWriter1L;

/**
 * {@code NetLink1} class primarily provided as a use case for the
 * {@code CNNLayers} component class.
 *
 * @author Sepehr Hooshiari
 *
 */
public class NetLink1 {

    /**
     * Representation of {@code this}.
     */
    private CNNLayers layers;

    /**
     * In case input values become too large.
     */
    private double scalar;

    /**
     * Constructor for {@code this}.
     *
     * @param layers
     *
     * @param scalar
     */
    public NetLink1(CNNLayers layers, double scalar) {
        this.layers = layers;
        this.scalar = scalar;
    }

    /**
     * Build the layers of the CNN.
     */
    public void build() {
        int i = 0;
        final int x = 3;
        while (i < x) {
            this.layers.addLayer(new CNNLayers1());
            i++;
        }
    }

    /**
     * Adds the elements of {@code y} to the elements of {@code x} and returns a
     * new array containing the sums.
     *
     * @param x
     *
     * @param y
     *
     * @return the array of sums
     */
    public double[] addVector(double[] x, double[] y) {
        double[] output = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            output[i] = x[i] + y[i];
        }
        return output;
    }

    /**
     * Multiplies the elements in {@code x} by the {@code scalar} and returns a
     * new array of the multiplied values.
     *
     * @param x
     *
     * @param scalar
     *
     * @return the array of products
     */
    public double[] multiplyVector(double[] x, double scalar) {
        double[] output = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            output[i] = x[i] * scalar;
        }
        return output;
    }

    /**
     * Returns an error list corresponding to the given data.
     *
     * @param outputs
     *
     * @param answer
     *
     * @return the error list
     */
    public double[] errorList(double[] outputs, int answer) {
        int classes = outputs.length;
        double[] expected = new double[classes];
        expected[answer] = 1;
        return this.addVector(outputs, this.multiplyVector(expected, -1));
    }

    /**
     * Gets the maximum value which corresponds to the guess of the label.
     *
     * @param inputs
     *
     * @return the max
     */
    public int getMax(double[] inputs) {
        double max = 0;
        int index = 0;
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] >= max) {
                max = inputs[i];
                index = i;
            }
        }
        return index;
    }

    /**
     * Uses the {@code CNNLayers} guess function to test the data.
     *
     * @param images
     *
     * @return the initial margin of error
     */
    public float test(List<double[][]> images) {
        int correct = 0;
        for (double[][] sig : images) {
            int guess = this.layers.guess(this.layers.getNext());
            if (guess == this.layers.getLabel()) {
                correct++;
            }
        }
        return ((float) correct / images.size());
    }

    /**
     * Trains the CNN to reduce margin of error.
     *
     * @param images
     *
     */
    public void train(List<double[][]> images) {
        for (double[][] sig : images) {
            List<double[][]> inputs = new ArrayList<>();
            inputs.add(this.layers.multiplyMatrix(this.layers.getData(),
                    (1.0 / this.scalar)));
            double[] out = this.layers.outputFromList(inputs);
            double[] lossPerOut = this.errorList(out, this.layers.getLabel());
            if (this.layers.getNext() != null) {
                this.layers.getNext().backPropArray(lossPerOut);
            }
        }
    }

    /**
     * Main method.
     *
     * @param args
     */
    public static void main(String[] args) {
        SimpleReader in = new SimpleReader1L();
        SimpleWriter out = new SimpleWriter1L();
        final double sFactor = 200 * 100;
        CNNLayers layers = new CNNLayers1();
        NetLink1 builder = new NetLink1(layers, sFactor);
        List<double[][]> images = layers.analyze("data/mnist_test.csv");
        List<double[][]> trainedImages = layers.analyze("data/mnist_train.csv");

        out.println("Testing images size: " + images.size());
        out.println("Training images size: " + trainedImages.size());
        out.println("Loading... ");

        builder.build();

        // success rate
        double sr = builder.test(images);
        out.println("Success rate prior to training: " + sr);

        int epochs = 2;

        for (int i = 0; i < epochs; i++) {
            shuffle(trainedImages);
            builder.train(trainedImages);
            sr = builder.test(images);
            out.println("Success rate after epoch " + i + ": " + sr);
        }

        /*
         * Close input and output streams
         */
        in.close();
        out.close();
    }
}
