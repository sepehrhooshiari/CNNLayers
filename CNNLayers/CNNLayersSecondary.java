import java.util.ArrayList;
import java.util.List;

/**
 * Layered implementations of secondary methods for {@code CNNLayers}.
 *
 * @author Sepehr Hooshiari
 *
 */
public abstract class CNNLayersSecondary implements CNNLayers {

    /**
     * Common methods (from Object). -------------------------------------------
     */

    // CHECKSTYLE: ALLOW THIS METHOD TO BE OVERRIDDEN
    @Override
    public String toString() {
        String s = this.getLabel() + ", \n";
        for (int i = 0; i < this.getData().length; i++) {
            for (int j = 0; j < this.getData()[0].length; j++) {
                s += this.getData()[i][j] + ", ";
            }
            s += "\n";
        }
        return s;
    }

    // CHECKSTYLE: ALLOW THIS METHOD TO BE OVERRIDDEN
    @Override
    public boolean equals(Object obj) {
        assert this != null : "Violation of : this is non-null";
        assert obj != null : "Violation of : argument is non-null";
        boolean check = false;
        if (obj.getClass() == this.getClass()
                && obj.hashCode() == this.hashCode()) {
            check = true;
        }
        return check;
    }

    // CHECKSTYLE: ALLOW THIS METHOD TO BE OVERRIDDEN
    @Override
    public int hashCode() {
        return this.inputLength() * this.outputLength()
                + (int) this.getLearnRate();
    }

    /**
     * Other non-kernel methods. -----------------------------------------------
     */

    // CHECKSTYLE: ALLOW THIS METHOD TO BE OVERRIDDEN
    @Override
    public double[] outputFromList(List<double[][]> input) {
        double[] vector = this.toArray(input);
        return this.outputFromArray(vector);
    }

    // CHECKSTYLE: ALLOW THIS METHOD TO BE OVERRIDDEN
    @Override
    public double[] outputFromArray(double[] input) {
        this.setNext(this.getNext());
        double[] pass = this.forwardPass(input);
        if (this.getNext().inputLength() != 0) {
            pass = this.outputFromArray(pass);
        }
        return pass;
    }

    // CHECKSTYLE: ALLOW THIS METHOD TO BE OVERRIDDEN
    @Override
    public void backPropList(List<double[][]> deriv) {
        double[] vector = this.toArray(deriv);
        this.backPropArray(vector);
    }

    // CHECKSTYLE: ALLOW THIS METHOD TO BE OVERRIDDEN
    @Override
    public void backPropArray(double[] deriv) {
        double sigDeriv;
        double weightDeriv;
        double outWeight;
        double loss;
        double[] prevLayer = new double[this.inputLength()];
        double[][] optWeights = this.getWeights();

        for (int i = 0; i < this.inputLength(); i++) {
            double prevSum = 0;
            for (int j = 0; j < this.outputLength(); j++) {
                sigDeriv = this.sigmoidPrime(this.getOutputs()[j]);
                weightDeriv = this.getInputs()[i];
                outWeight = this.getWeights()[i][j];
                loss = deriv[j] * sigDeriv * weightDeriv;
                optWeights[i][j] -= loss * this.getLearnRate();
                prevSum += deriv[j] * sigDeriv * outWeight;
            }
            prevLayer[i] = prevSum;
        }
        this.setWeightsIn(optWeights);

        if (prevLayer.length != 0) {
            this.getPrevious().backPropArray(prevLayer);
        }
    }

    // CHECKSTYLE: ALLOW THIS METHOD TO BE OVERRIDDEN
    @Override
    public int guess(CNNLayers sig) {
        int index = 0;
        final double scalar = 200 * 100;
        List<double[][]> inputs = new ArrayList<>();
        inputs.add(this.multiplyMatrix(sig.getData(), (1.0 / scalar)));
        if (this.getLayers().size() > 0) {
            double[] out = this.getLayers().get(0).outputFromList(inputs);
            double max = 0;
            for (int i = 0; i < out.length; i++) {
                if (out[i] >= max) {
                    max = out[i];
                    index = i;
                }
            }
        }
        return index;
    }

}
