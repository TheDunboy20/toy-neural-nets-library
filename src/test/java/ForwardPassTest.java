import msmarik.Layer;
import msmarik.Net;
import msmarik.Perceptron;
import msmarik.activations.Activations;
import msmarik.losses.Losses;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class ForwardPassTest {
    @Test
    public void testForwardPass() {
        final Net net = new Net(Losses.MSE(), 1);

        // Hidden Layer 1
        Perceptron h1 = new Perceptron(Activations.linear(), new double[]{0.1, 0.2}, 0, "Layer-1-Perceptron-1");
        Perceptron h2 = new Perceptron(Activations.linear(), new double[]{0.3, 0.4}, 0, "Layer-1-Perceptron-2");
        Layer hl1 = new Layer(new Perceptron[]{h1, h2});

        // Hidden Layer 2
        Perceptron h3 = new Perceptron(Activations.linear(), new double[]{0.5, 0.6}, 0, "Layer-2-Perceptron-1");
        Perceptron h4 = new Perceptron(Activations.linear(), new double[]{0.7, 0.8}, 0, "Layer-2-Perceptron-2");
        Layer hl2 = new Layer(new Perceptron[]{h3, h4});

        // Output Layer
        Perceptron h5 = new Perceptron(Activations.linear(), new double[]{0.9, 1.0}, 0, "Layer-3-Perceptron-1");
        Layer hl3 = new Layer(new Perceptron[]{h5});
        net.addLayers(hl1, hl2, hl3);

        double correctLabel = 1;

        double[] netOutput = net.forward(new double[]{1 , 2});
        double netLoss = net.calculateLoss(netOutput, new double[]{correctLabel});
        net.backpropagate(correctLabel);

        assert Arrays.equals(netOutput, new double[]{2.049});
        assert netLoss == 1.100401;
    }
}
