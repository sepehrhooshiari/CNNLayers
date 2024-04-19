import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

/**
 *
 * @author Sepehr Hooshiari
 *
 */
public class CNNLayers1Test {

    /**
     * Test constructor with no arguments.
     */
    @Test
    public void testNoArgsConstructor() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = new CNNLayers1();
        assertTrue(ins1.isValid());
        assertEquals(ins2, ins1);
    }

    /**
     * Test constructor with arguments.
     */
    @Test
    public void testConstructor() {
        final CNNLayers ins1 = new CNNLayers1(10, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(10, 10, 123, 0.1);
        assertTrue(ins1.isValid());
        assertEquals(ins2, ins1);
    }

    /**
     * Test standard method newInstance.
     */
    @Test
    public void testNewInstance() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = ins1.newInstance();
        assertEquals(ins2, ins1);
    }

    /**
     * Test standard method clear.
     */
    @Test
    public void testClear() {
        final CNNLayers ins1 = new CNNLayers1(10, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1();
        ins1.clear();
        assertEquals(ins2, ins1);
    }

    /**
     * Test standard method transferFrom.
     */
    @Test
    public void testTransferFrom() {
        final CNNLayers ins1 = new CNNLayers1(10, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(10, 10, 123, 0.1);
        final CNNLayers ins3 = new CNNLayers1();
        final CNNLayers ins4 = new CNNLayers1();
        ins3.transferFrom(ins2);
        assertEquals(ins1, ins3);
        assertEquals(ins4, ins2);
    }

    /**
     * Test if addLayer links to existing layers properly.
     */
    @Test
    public void testAddLayer() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = new CNNLayers1();
        ins1.addLayer(ins2);
        assertEquals(ins1.getNext(), ins2);
        assertEquals(ins1, ins2.getPrevious());
    }

    /**
     * Test if getNext returns an existing value.
     */
    @Test
    public void testGetNext() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = new CNNLayers1();
        assertTrue(ins1.getNext().isValid());
        assertEquals(ins2, ins1);
    }

    /**
     * Test if setNext follows existing convention.
     */
    @Test
    public void testSetNext() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = new CNNLayers1(10, 10, 123, 0.1);
        ins1.setNext(ins2);
        assertEquals(ins2, ins1.getNext());
    }

    /**
     * Test if getPrevious returns an existing value.
     */
    @Test
    public void testGetPrevious() {
        final CNNLayers ins1 = new CNNLayers1(10, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(10, 10, 123, 0.1);
        assertTrue(ins1.getPrevious().isValid());
        assertEquals(ins2, ins1);
    }

    /**
     * Test if setPrevious follows existing convention.
     */
    @Test
    public void testSetPrevious() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = new CNNLayers1(10, 10, 123, 0.1);
        ins1.setPrevious(ins2);
        assertEquals(ins2, ins1.getPrevious());
    }

    /**
     * Test if getData stores and returns data properly.
     */
    @Test
    public void testGetData() {
        final CNNLayers ins = new CNNLayers1(10, 10, 123, 0.1);
        // this should always be the length of the data
        final int x = 28;
        assertEquals(x, ins.getData().length);
    }

    /**
     * Test if getLabel stores and returns label properly.
     */
    @Test
    public void testGetLabel() {
        final CNNLayers ins = new CNNLayers1(10, 10, 123, 0.1);
        final int n = 9;
        // the label should be a positive, single-digit integer
        assertTrue(ins.getLabel() >= 0 || ins.getLabel() <= n);
    }

    /**
     * Test if inputLength returns the proper input length.
     */
    @Test
    public void testInputLength() {
        final CNNLayers ins1 = new CNNLayers1(1, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(1, 10, 123, 0.1);
        assertEquals(1, ins1.inputLength());
        assertEquals(ins2, ins1);
    }

    /**
     * Test if outputLength returns the proper output length.
     */
    @Test
    public void testOutputLength() {
        final CNNLayers ins1 = new CNNLayers1(10, 1, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(10, 1, 123, 0.1);
        assertEquals(1, ins1.outputLength());
        assertEquals(ins2, ins1);
    }

    /**
     * Test getInputs before the call to forwardPass (this means that the array
     * of inputs should be empty since the CNN hasn't started passing inputs
     * between layers yet).
     */
    @Test
    public void testInputsBeforeFP() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        int sum = 0;
        for (int i = 0; i < ins1.getInputs().length; i++) {
            sum += ins1.getInputs()[i];
        }
        assertEquals(0, sum);
        assertEquals(ins2, ins1);
    }

    /**
     * Test getInputs after the call to forwardPass (this means that the array
     * of inputs shouldn't be empty since the CNN has started passing inputs
     * between layers).
     */
    @Test
    public void testInputsAfterFP() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        List<double[][]> images = ins1.analyze("data/mnist_test.csv");
        double[] inputs = ins1.toArray(images);
        ins1.forwardPass(inputs);
        int sum = 0;
        for (int i = 0; i < ins1.getInputs().length; i++) {
            sum += ins1.getInputs()[i];
        }
        assertTrue(sum != 0);
        assertEquals(ins2, ins1);
    }

    /**
     * Test getOutputs before the call to forwardPass (this means that the array
     * of outputs should be empty since the CNN hasn't started passing outputs
     * between layers yet).
     */
    @Test
    public void testOutputsBeforeFP() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        int sum = 0;
        for (int i = 0; i < ins1.getOutputs().length; i++) {
            sum += ins1.getOutputs()[i];
        }
        assertEquals(0, sum);
        assertEquals(ins2, ins1);
    }

    /**
     * Test getOutputs after the call to forwardPass (this means that the array
     * of outputs shouldn't be empty since the CNN has started passing outputs
     * between layers).
     */
    @Test
    public void testOutputsAfterFP() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        List<double[][]> images = ins1.analyze("data/mnist_test.csv");
        double[] inputs = ins1.toArray(images);
        ins1.forwardPass(inputs);
        int sum = 0;
        for (int i = 0; i < ins1.getOutputs().length; i++) {
            sum += ins1.getOutputs()[i];
        }
        assertTrue(sum != 0);
        assertEquals(ins2, ins1);
    }

    /**
     * Test if getWeights returns a non-empty, randomly distributed matrix of
     * weights.
     */
    @Test
    public void testGetWeights() {
        final CNNLayers ins1 = new CNNLayers1(10, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(10, 10, 123, 0.1);
        double[][] weights = ins1.getWeights();
        int sum = 0;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights.length; j++) {
                sum += weights[i][j];
            }
        }
        assertTrue(sum != 0);
        assertEquals(ins2, ins1);
    }

    /**
     * Test if getLearnRate returns a reasonable learning rate value for the
     * CNN.
     */
    @Test
    public void testGetLearnRate() {
        final CNNLayers ins1 = new CNNLayers1(10, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(10, 10, 123, 0.1);
        final double exp = 0.1;
        final double delta = 0.0001;
        assertEquals(exp, ins1.getLearnRate(), delta);
        assertEquals(ins2, ins1);
    }

    /**
     * Test isValid with different instantiations.
     */
    @Test
    public void testIsValid() {
        final CNNLayers ins1 = new CNNLayers1(10, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1();
        final CNNLayers ins3 = ins1.newInstance();
        assertTrue(ins1.isValid());
        assertTrue(ins2.isValid());
        assertTrue(ins3.isValid());
    }

    /**
     * Test if toArray returns the proper converted array.
     */
    @Test
    public void testToArray() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = new CNNLayers1();
        List<double[][]> matrices = new ArrayList<>();
        final int x = 5;
        final int y = 20;
        for (int i = 0; i < x; i++) {
            matrices.add(new double[2][2]);
        }
        double[] exp = new double[y];
        double[] actual = ins1.toArray(matrices);
        assertTrue(Arrays.equals(exp, actual));
        assertEquals(ins2, ins1);
    }

    /**
     * Test if toMatrix returns the proper converted matrices.
     */
    @Test
    public void testToMatrix() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = new CNNLayers1();
        final int x = 20;
        double[] arr = new double[x];
        List<double[][]> exp = new ArrayList<>();
        final int y = 5;
        for (int i = 0; i < y; i++) {
            exp.add(new double[2][2]);
        }
        List<double[][]> actual = ins1.toMatrix(arr, exp.size(),
                exp.get(0).length, exp.get(0)[0].length);
        assertEquals(exp.get(1)[1][1], actual.get(1)[1][1], 1);
        assertEquals(ins2, ins1);
    }

    /**
     * Test if forwardPass returns the proper output array.
     */
    @Test
    public void testForwardPass() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        List<double[][]> images = ins1.analyze("data/mnist_test.csv");
        double[] inputs = ins1.toArray(images);
        double[] out = ins1.forwardPass(inputs);
        int sum = 0;
        for (int i = 0; i < out.length; i++) {
            sum += out[i];
        }
        assertTrue(sum != 0);
        assertEquals(ins2, ins1);
    }

    /**
     * Test if sigmoidPrime returns the proper derivative of the Sigmoid
     * function.
     */
    @Test
    public void testSigmoidPrime() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        final double x = 8;
        final double delta = 0.000001;
        double actual = ins1.sigmoidPrime(x);
        final double exp = 0.000335;
        assertEquals(exp, actual, delta);
        assertEquals(ins2, ins1);
    }

}
