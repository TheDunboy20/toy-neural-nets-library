import msmarik.Layer;
import msmarik.Net;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class BackpropagationTest implements FixedNNTest{

    @Test
    void forwardOutputsAreImmutable() {
        Net net = buildFixedNet();
        double[] input = {1, 2};

        // Capture forward outputs
        double[] outputBefore = net.forward(input).clone();

        // Run backpropagation
        net.backpropagate(1.0);

        double[] outputAfter = net.forward(input).clone();

        // Ensure forward outputs did not change
        assertArrayEquals(outputBefore, outputAfter, 1e-12);
    }

    @Test
    void layerErrorSignals() {
        Net net = buildFixedNet();
        double[] input = {1, 2};
        double target = 1.0;

        // Forward pass
        net.forward(input);

        // Backpropagation
        net.backpropagate(target);

        // Capture actual error signals
        Layer l1 = net.getLayer(0);
        Layer l2 = net.getLayer(1);
        Layer l3 = net.getLayer(2);

        double[] l1Error = new double[l1.getPerceptrons().length];
        for (int i = 0; i < l1Error.length; i++)
            l1Error[i] = l1.getPerceptrons()[i].getErrorSignal();

        double[] l2Error = new double[l2.getPerceptrons().length];
        for (int i = 0; i < l2Error.length; i++)
            l2Error[i] = l2.getPerceptrons()[i].getErrorSignal();

        double[] l3Error = new double[l3.getPerceptrons().length];
        for (int i = 0; i < l3Error.length; i++)
            l3Error[i] = l3.getPerceptrons()[i].getErrorSignal();

        // === Assertions ===
        double[] expectedL1 = {2.4127, 2.8113};
        double[] expectedL2 = {1.8882, 2.098};
        double[] expectedL3 = {2.098};

        for (int i = 0; i < expectedL1.length; i++)
            assertEquals(expectedL1[i], l1Error[i], 1e-4);

        for (int i = 0; i < expectedL2.length; i++)
            assertEquals(expectedL2[i], l2Error[i], 1e-4);

        for (int i = 0; i < expectedL3.length; i++)
            assertEquals(expectedL3[i], l3Error[i], 1e-4);
    }
}
