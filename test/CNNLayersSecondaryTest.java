import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Test;

/**
 *
 * @author Sepehr Hooshiari
 *
 */
public class CNNLayersSecondaryTest {

    /**
     * Test object method equals.
     */
    @Test
    public void testEquals() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        assertTrue(ins2.equals(ins1));
    }

    /**
     * Test object method hashCode.
     */
    @Test
    public void testHashCode() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        assertEquals(ins2.hashCode(), ins1.hashCode());
        assertEquals(ins2, ins1);
    }

    /**
     * Test secondary method outputFromList with many inputs.
     */
    @Test
    public void testOutFromListMany() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        List<double[][]> images = ins1.analyze("data/mnist_test.csv");
        double[] arr = ins1.outputFromList(images);
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        assertTrue(sum != 0);
        assertEquals(ins2, ins1);
    }

    /**
     * Test secondary method outputFromList with no inputs.
     */
    @Test
    public void testOutFromListNone() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = new CNNLayers1();
        List<double[][]> images = ins1.analyze("data/mnist_test.csv");
        double[] arr = ins1.outputFromList(images);
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        assertTrue(sum == 0);
        assertEquals(ins2, ins1);
    }

    /**
     * Test secondary method outputFromArray with many inputs.
     */
    @Test
    public void testOutFromArrayMany() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        List<double[][]> images = ins1.analyze("data/mnist_test.csv");
        double[] arr = ins1.outputFromArray(ins1.toArray(images));
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        assertTrue(sum != 0);
        assertEquals(ins2, ins1);
    }

    /**
     * Test secondary method outputFromArray with no inputs.
     */
    @Test
    public void testOutFromArrayNone() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = new CNNLayers1();
        List<double[][]> images = ins1.analyze("data/mnist_test.csv");
        double[] arr = ins1.outputFromArray(ins1.toArray(images));
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        assertTrue(sum == 0);
        assertEquals(ins2, ins1);
    }

    /**
     * Test secondary method backPropArray with many inputs.
     */
    @Test
    public void testBackPropArrayMany() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        List<double[][]> images = ins1.analyze("data/mnist_train.csv");
        double[] input = ins1.outputFromList(images);
        double sum1 = 0;
        for (int i = 0; i < ins1.getWeights().length; i++) {
            for (int j = 0; j < ins1.getWeights()[0].length; j++) {
                sum1 += ins1.getWeights()[i][j];
            }
        }
        // back propagation should optimize the weights
        ins1.backPropArray(input);
        double sum2 = 0;
        for (int k = 0; k < ins1.getWeights().length; k++) {
            for (int l = 0; l < ins1.getWeights()[0].length; l++) {
                sum2 += ins1.getWeights()[k][l];
            }
        }
        assertTrue(sum1 > sum2);
        assertEquals(ins2, ins1);
    }

    /**
     * Test secondary method backPropArray with no inputs.
     */
    @Test
    public void testBackPropArrayNone() {
        final CNNLayers ins1 = new CNNLayers1();
        final CNNLayers ins2 = new CNNLayers1();
        List<double[][]> images = ins1.analyze("data/mnist_train.csv");
        double[] input = ins1.outputFromList(images);
        double sum1 = 0;
        for (int i = 0; i < ins1.getWeights().length; i++) {
            for (int j = 0; j < ins1.getWeights()[0].length; j++) {
                sum1 += ins1.getWeights()[i][j];
            }
        }
        ins1.backPropArray(input);
        double sum2 = 0;
        for (int k = 0; k < ins1.getWeights().length; k++) {
            for (int l = 0; l < ins1.getWeights()[0].length; l++) {
                sum2 += ins1.getWeights()[k][l];
            }
        }
        assertTrue(sum1 == sum2);
        assertEquals(ins2, ins1);
    }

    /**
     * Test secondary method backPropList with many inputs.
     */
    @Test
    public void testBackPropListMany() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        List<double[][]> images = ins1.analyze("data/mnist_train.csv");
        double[] input = ins1.outputFromList(images);
        double sum1 = 0;
        for (int i = 0; i < ins1.getWeights().length; i++) {
            for (int j = 0; j < ins1.getWeights()[0].length; j++) {
                sum1 += ins1.getWeights()[i][j];
            }
        }
        // back propagation should optimize the weights
        ins1.backPropList(ins1.toMatrix(input, ins1.outputLength(), 1, 1));
        double sum2 = 0;
        for (int k = 0; k < ins1.getWeights().length; k++) {
            for (int l = 0; l < ins1.getWeights()[0].length; l++) {
                sum2 += ins1.getWeights()[k][l];
            }
        }
        assertTrue(sum1 > sum2);
        assertEquals(ins2, ins1);
    }

    /**
     * Test secondary method backPropList with no inputs.
     */
    @Test
    public void testBackPropListNone() {
        final CNNLayers ins1 = new CNNLayers1(0, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(0, 10, 123, 0.1);
        List<double[][]> images = ins1.analyze("data/mnist_train.csv");
        double[] input = ins1.outputFromList(images);
        double sum1 = 0;
        for (int i = 0; i < ins1.getWeights().length; i++) {
            for (int j = 0; j < ins1.getWeights()[0].length; j++) {
                sum1 += ins1.getWeights()[i][j];
            }
        }
        ins1.backPropList(ins1.toMatrix(input, ins1.outputLength(), 1, 1));
        double sum2 = 0;
        for (int k = 0; k < ins1.getWeights().length; k++) {
            for (int l = 0; l < ins1.getWeights()[0].length; l++) {
                sum2 += ins1.getWeights()[k][l];
            }
        }
        assertTrue(sum1 == sum2);
        assertEquals(ins2, ins1);
    }

    /**
     * Test secondary method guess for proper functionality.
     */
    @Test
    public void testGuess() {
        final CNNLayers ins1 = new CNNLayers1(300, 10, 123, 0.1);
        final CNNLayers ins2 = new CNNLayers1(300, 10, 123, 0.1);
        final int z = 5;
        int x = ins1.guess(ins2);
        ins1.addLayer(ins2);
        assertTrue(x - ins1.getLabel() < z);
        assertEquals(ins2, ins1);
    }

}
