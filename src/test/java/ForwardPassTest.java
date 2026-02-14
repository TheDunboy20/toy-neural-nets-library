import msmarik.Layer;
import msmarik.Net;
import msmarik.Perceptron;
import msmarik.activations.Activations;
import msmarik.losses.Losses;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class ForwardPassTest {
    @Test
    public void forward() {
        Net net = buildFixedNet();

        Layer l1 = net.getLayer(0);
        Layer l2 = net.getLayer(1);
        Layer l3 = net.getLayer(2);

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

    @Test
    void forward_isDeterministic() {
        Net net = buildFixedNet();

        double[] input = {1, 2};

        double[] out1 = net.forward(input);
        double[] out2 = net.forward(input);
        double[] out3 = net.forward(input);

        assertArrayEquals(out1, out2, 1e-12);
        assertArrayEquals(out2, out3, 1e-12);
    }

    @Test
    void forward_doesNotChangeWeightsOrBiases() {
        Net net = buildFixedNet();

        // capture original weights
        double[][][] originalWeights = netWeightsSnapshot(net);

        double[] input = {1, 2};
        net.forward(input);

        double[][][] afterWeights = netWeightsSnapshot(net);

        for (int l = 0; l < originalWeights.length; l++) {
            for (int p = 0; p < originalWeights[l].length; p++) {
                assertArrayEquals(
                        originalWeights[l][p],
                        afterWeights[l][p],
                        1e-12
                );
            }
        }
    }

    @Test
    void forward_outputShapeIsCorrect() {
        Net net = buildFixedNet();

        double[] out = net.forward(new double[]{1, 2});

        assertNotNull(out);
        assertEquals(1, out.length);
    }

    private Net buildFixedNet() {
        Net net = new Net(Losses.MSE(), 1);

        // Layer 1
        Perceptron h1 = new Perceptron(Activations.linear(), new double[]{0.1, 0.2}, 0, "h1");
        Perceptron h2 = new Perceptron(Activations.linear(), new double[]{0.3, 0.4}, 0, "h2");
        Layer l1 = new Layer(new Perceptron[]{h1, h2});

        // Layer 2
        Perceptron h3 = new Perceptron(Activations.linear(), new double[]{0.5, 0.6}, 0, "h3");
        Perceptron h4 = new Perceptron(Activations.linear(), new double[]{0.7, 0.8}, 0, "h4");
        Layer l2 = new Layer(new Perceptron[]{h3, h4});

        // Output
        Perceptron h5 = new Perceptron(Activations.linear(), new double[]{0.9, 1.0}, 0, "h5");
        Layer l3 = new Layer(new Perceptron[]{h5});

        net.addLayers(l1, l2, l3);
        return net;
    }

    private double[][][] netWeightsSnapshot(Net net) {
        List<Layer> layers = net.getLayers();
        double[][][] snapshot = new double[layers.size()][][];

        for (int i = 0; i < layers.size(); i++) {
            Perceptron[] ps = layers.get(i).getPerceptrons();
            snapshot[i] = new double[ps.length][];

            for (int j = 0; j < ps.length; j++) {
                snapshot[i][j] = ps[j].getWeights().clone();
            }
        }
        return snapshot;
    }


}
