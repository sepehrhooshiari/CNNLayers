import java.util.List;

/**
 * {@code CNNLayersKernel} enhanced with secondary methods. (Note: by
 * package-wide convention, all references are non-null.)
 *
 * @author Sepehr Hooshiari
 *
 * @mathsubtypes <pre>
 * CNN_LAYERS_MODEL is (
 *   matrix of doubles
 *  )
 * </pre>
 * @mathmodel type CNNLayers is modeled by CNN_LAYERS_MODEL
 * @initially {@code
 * ():
 *  ensures
 *   this = (0, 0, 0, 0, [0][0])
 * }
 */
public interface CNNLayers extends CNNLayersKernel {

    /**
     * During forward passes between layers, outputs a {@code double[]} from a
     * given input of matrices to be passed onto the next layer.
     *
     * @param input
     *            the {@code List} of matrices
     *
     * @ensures <pre> input = #input </pre>
     *
     * @return the output array
     */
    double[] outputFromList(List<double[][]> input);

    /**
     * During forward passes between layers, outputs a {@code double[]} from a
     * given input array to be passed onto the next layer.
     *
     * @param input
     *            the {@code double[]} input
     *
     * @ensures <pre> input = #input </pre>
     *
     * @return the output array
     */
    double[] outputFromArray(double[] input);

    /**
     * During back propagation between layers, updates weights/filters/maximums
     * when the current layer's loss with respect to outputs is a {@code List}
     * of matrices.
     *
     * @param deriv
     *            loss with respect to outputs
     *
     * @ensures <pre> deriv = #deriv </pre>
     */
    void backPropList(List<double[][]> deriv);

    /**
     * During back propagation between layers, updates weights/filters/maximums
     * when the current layer's loss with respect to outputs is a
     * {@code double[]}.
     *
     * @param deriv
     *            loss with respect to outputs
     *
     * @ensures <pre> deriv = #deriv </pre>
     */
    void backPropArray(double[] deriv);

    /**
     * Function for the CNN to make an initial guess of what the image depicts.
     *
     * @param sig
     *            the current layer of the CNN
     *
     * @return the initial guess of the image's label
     */
    int guess(CNNLayers sig);

}
