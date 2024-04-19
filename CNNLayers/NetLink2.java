import static java.util.Collections.shuffle;

import java.util.List;

import components.simplereader.SimpleReader;
import components.simplereader.SimpleReader1L;
import components.simplewriter.SimpleWriter;
import components.simplewriter.SimpleWriter1L;

/**
 * {@code NetLink2} class primarily provided as a use case for the
 * {@code CNNLayers} component class.
 *
 * @author Sepehr Hooshiari
 *
 */
public final class NetLink2 {

    /**
     * No argument constructor--private to prevent instantiation.
     */
    private NetLink2() {
    }

    /**
     * Main method.
     *
     * @param args
     */
    public static void main(String[] args) {
        SimpleReader in = new SimpleReader1L();
        SimpleWriter out = new SimpleWriter1L();
        final double sFactor = 300 * 100;
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

        final int epochs = 5;

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
