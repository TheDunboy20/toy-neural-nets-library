import msmarik.Layer;
import msmarik.Net;
import msmarik.Perceptron;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class WeightBiasUpdateTest implements FixedNNTest{

    @Test
    void singleUpdateCorrectness() {
        Net net = buildFixedNet();
        double[] input = {1, 2};
        double target = 1.0;

        net.forward(input);
        net.backpropagate(target);
        net.updateWeightsAndBiases(0.001);

        // output layer
        double[] expectedWeightsL3 = {0.8980908, 0.9974195};
        Perceptron outNeuron = net.getLayer(2).getPerceptrons()[0];
        double[] updatedWeightsL3 = outNeuron.getWeights();
        for (int i = 0; i < expectedWeightsL3.length; i++) {
            assertEquals(expectedWeightsL3[i], updatedWeightsL3[i], 1e-4);
        }

        double[][] expectedWeightsL2 = {
                {0.499056, 0.597923},  // h3
                {0.699051, 0.797692}   // h4
        };
        checkHiddenLayerWeights(net.getLayer(1), expectedWeightsL2);

        double[][] expectedWeightsL1 = {
                {0.0975887, 0.1951746},  // h1
                {0.2971887, 0.394377}    // h2
        };
        checkHiddenLayerWeights(net.getLayer(0), expectedWeightsL1);
    }

    private void checkHiddenLayerWeights(Layer hiddenLayer, double[][] expectedWeights) {
        Perceptron[] perceptrons = hiddenLayer.getPerceptrons();
        for (int p = 0; p < perceptrons.length; p++) {
            double[] weights = perceptrons[p].getWeights();
            for (int i = 0; i < weights.length; i++) {
                assertEquals(expectedWeights[p][i], weights[i], 1e-4,
                        "Mismatch at perceptron " + p + ", weight " + i);
            }
        }
    }

}
