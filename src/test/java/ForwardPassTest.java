import msmarik.Layer;
import msmarik.Net;
import msmarik.Perceptron;
import msmarik.activations.Activations;
import msmarik.losses.Losses;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ForwardPassTest {
    @Test
    public void testForwardPass() {
        Net net = new Net(Losses.MSE(), 1);

        // ===== Layer 1 =====
        Perceptron h1 = new Perceptron(Activations.linear(), new double[]{0.1, 0.2}, 0, "h1");
        Perceptron h2 = new Perceptron(Activations.linear(), new double[]{0.3, 0.4}, 0, "h2");
        Layer l1 = new Layer(new Perceptron[]{h1, h2});

        // ===== Layer 2 =====
        Perceptron h3 = new Perceptron(Activations.linear(), new double[]{0.5, 0.6}, 0, "h3");
        Perceptron h4 = new Perceptron(Activations.linear(), new double[]{0.7, 0.8}, 0, "h4");
        Layer l2 = new Layer(new Perceptron[]{h3, h4});

        // ===== Output layer =====
        Perceptron h5 = new Perceptron(Activations.linear(), new double[]{0.9, 1.0}, 0, "h5");
        Layer l3 = new Layer(new Perceptron[]{h5});

        net.addLayers(l1, l2, l3);

        double[] input = {1, 2};

        // ---- forward manually layer by layer ----
        double[] out1 = l1.forward(input);
        double[] out2 = l2.forward(out1);
        double[] out3 = l3.forward(out2);

        // ===== assertions =====

        // layer 1
        assertEquals(0.5, out1[0], 1e-9);
        assertEquals(1.1, out1[1], 1e-9);

        // layer 2
        assertEquals(0.91, out2[0], 1e-9);
        assertEquals(1.23, out2[1], 1e-9);

        // final output
        assertEquals(2.049, out3[0], 1e-9);

        double loss = net.calculateLoss(out3, new double[]{1});
        assertEquals(1.100401, loss, 1e-9);
    }
}
