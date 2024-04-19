import java.util.List;

import components.standard.Standard;

/**
 * Convolutional Neural Network (CNN) Layers kernel component with primary
 * methods. (Note: by package-wide convention, all references are non-null.)
 *
 * @author Sepehr Hooshiari
 *
 * @mathsubtypes <pre>
 * CNN_LAYERS_MODEL is (
 *   matrix of doubles
 *  )
 * </pre>
 * @mathmodel type CNNLayersKernel is modeled by CNN_LAYERS_MODEL
 * @initially {@code
 * ():
 *  ensures
 *   this = (0, 0, 0, 0, [0][0])
 * }
 */
public interface CNNLayersKernel extends Standard<CNNLayers> {

    /**
     * Returns the next layer after {@code this}.
     *
     * @requires <pre> next layer exists </pre>
     *
     * @return next layer
     */
    CNNLayers getNext();

    /**
     * Sets the next layer of {@code this} as {@code nextLayer}.
     *
     * @requires <pre> {@code nextLayer} != null </pre>
     *
     * @param nextLayer
     *            the next layer
     */
    void setNext(CNNLayers nextLayer);

    /**
     * Returns the previous layer before {@code this}.
     *
     * @requires <pre> previous layer exists </pre>
     *
     * @return previous layer
     */
    CNNLayers getPrevious();

    /**
     * Sets the previous layer of {@code this} as {@code previousLayer}.
     *
     * @requires <pre> {@code previousLayer} != null </pre>
     *
     * @param previousLayer
     *            the previous layer
     */
    void setPrevious(CNNLayers previousLayer);

    /**
     * Returns the input data corresponding to an image as a {@code double[][]}.
     *
     * @return input data
     */
    double[][] getData();

    /**
     * Returns the label of the input data which indicates what number the image
     * represents.
     *
     * @return data label
     */
    int getLabel();

    /**
     * Returns the {@code List} of layers that the CNN should have.
     *
     * @return the CNN's layers
     */
    List<CNNLayers> getLayers();

    /**
     * Copies the fields of {@code source} to {@code this} without changing the
     * state of {@code source}.
     *
     * @param source
     *            the {@code CNNLayers} object to copy from
     */
    void copyFrom(CNNLayers source);

    /**
     * Multiplies the elements in {@code x} by the provided {@code scalar}.
     *
     * @param x
     *            the {@code double[][]} matrix
     * @param scalar
     *            the scalar to multiply by
     * @updates x
     *
     * @return the updated matrix
     */
    double[][] multiplyMatrix(double[][] x, double scalar);

    /**
     * Stores the data from the given file into the data of {@code this} and the
     * data's corresponding label is stored as well.
     *
     * @param s
     *            the name of the input file
     *
     * @return the {@code List} of translated images
     */
    List<double[][]> analyze(String s);

    /**
     * Adds the input {@code layer} to {@code this}.
     *
     * @param layer
     *            the layer to be added
     *
     * @ensures <pre> {@code layer} = next layer or previous layer </pre>
     */
    void addLayer(CNNLayers layer);

    /**
     * Returns the number of inputs to this layer.
     *
     * @return input length
     */
    int inputLength();

    /**
     * Returns the number of outputs from this layer.
     *
     * @return output length
     */
    int outputLength();

    /**
     * Returns the {@code double[]} of inputs to this layer.
     *
     * @return array of inputs
     */
    double[] getInputs();

    /**
     * Returns the {@code double[]} of outputs from this layer.
     *
     * @return array of outputs
     */
    double[] getOutputs();

    /**
     * Returns the matrix of weights which correspond to the inputs of this
     * layer.
     *
     * @return matrix of weights
     */
    double[][] getWeights();

    /**
     * Updates the weights of {@code this} manually.
     *
     * @ensures <pre> this.weights.length = this.inLength and
     * this.weights[0].length = this.outLength </pre>
     *
     * @param weights
     *            the weights to be stored
     */
    void setWeightsIn(double[][] weights);

    /**
     * Returns the rate at which the CNN should learn.
     *
     * @return learning rate
     */
    double getLearnRate();

    /**
     * Returns whether or not representation is non-null.
     *
     * @ensures <pre> this.getWeights() != null </pre>
     *
     * @return true if {@code double[][]} of weights is non-null
     */
    boolean isValid();

    /**
     * Converts the {@code List} input to a new {@code double[]}, returning the
     * new array of doubles.
     *
     * @param input
     *            the {@code List} of matrices
     *
     * @ensures <pre> input = #input and this.length = input.size() </pre>
     *
     * @return the converted array
     */
    double[] toArray(List<double[][]> input);

    /**
     * Converts the {@code double[]} input to a new {@code List}, returning the
     * new List of matrices.
     *
     * @param input
     *            the {@code double[]} array
     * @param length
     *            the number of matrices
     * @param rows
     *            number of rows in each matrix
     * @param columns
     *            number of columns in each matrix
     *
     * @requires <pre> input.length = length * rows * columns </pre>
     *
     * @ensures <pre> input = #input and this.size() = input.length </pre>
     *
     * @return the converted list of matrices
     */
    List<double[][]> toMatrix(double[] input, int length, int rows,
            int columns);

    /**
     * Multiplies each input in the fully connected layer by its corresponding
     * weight and returns the array of weighted outputs.
     *
     * @param input
     *            the input vector
     *
     * @ensures <pre> input = #input </pre>
     *
     * @return the weighted output vector
     */
    double[] forwardPass(double[] input);

    /**
     * Returns the derivative of the {@code double} input.
     *
     * @param input
     *            the Sigmoid activated weighted sum
     *
     * @ensures <pre> input = #input </pre>
     *
     * @return the derivative of the input
     */
    double sigmoidPrime(double input);

}
