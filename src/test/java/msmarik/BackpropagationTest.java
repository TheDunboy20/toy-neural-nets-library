package msmarik;

import msmarik.activations.Activations;
import msmarik.losses.Losses;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class BackpropagationTest implements FixedNNTest{

    @Test
    void forwardOutputsAreImmutable() {
        Net net = buildFixedNet();
        double[] input = {1, 2};

        double[] outputBefore = net.forward(input).clone();

        net.backpropagate(new double[] {1.0});

        double[] outputAfter = net.forward(input).clone();

        assertArrayEquals(outputBefore, outputAfter, 1e-12);
    }

    @Test
    void layerErrorSignals() {
        Net net = buildFixedNet();
        double[] input = {1, 2};
        double target = 1.0;

        net.forward(input);

        net.backpropagate(new double[] {target});

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

    @Test
    void weightGradients() {
        Net net = buildFixedNet();
        double[] input = {1, 2};
        double target = 1.0;

        net.forward(input);
        net.backpropagate(new double[] {target});

        Layer l1 = net.getLayer(0);
        Layer l2 = net.getLayer(1);
        Layer l3 = net.getLayer(2);

        this.gradientTestOutPutLayer(l3, new double[]{1.9092, 2.5805});
        this.testGradientHiddenLayer(l2, new double[][]{{0.9441, 2.0770}, {1.0490, 2.3078}});
        this.testGradientHiddenLayer(l1, new double[][]{{2.4127, 4.8254}, {2.8113, 5.6226}});
    }

    private void gradientTestOutPutLayer(Layer l3, double[] expectedGradients) {
        Perceptron outNeuron = l3.getPerceptrons()[0];
        double[] lastInput3 = outNeuron.getLastInput();
        double[] grad3 = new double[lastInput3.length];
        for (int i = 0; i < grad3.length; i++)
            grad3[i] = outNeuron.getErrorSignal() * lastInput3[i];

        for (int i = 0; i < grad3.length; i++)
            assertEquals(expectedGradients[i], grad3[i], 1e-4);
    }

    private void testGradientHiddenLayer(Layer hiddenLayer, double[][] expectedGradients) {
        Perceptron[] hiddenLayerPerceptrons = hiddenLayer.getPerceptrons();
        int perceptronInputLength = hiddenLayerPerceptrons[0].getLastInput().length;
        double[][] hiddenLayerGradients = new double[perceptronInputLength][perceptronInputLength];

        int perceptronIndex = 0;
        for (Perceptron perceptron : hiddenLayerPerceptrons) {
            double[] lastInput2 = perceptron.getLastInput();
            double[] perceptronGradient = new double[perceptronInputLength];

            for (int i = 0; i < perceptronGradient.length; i++) {
                double grad = perceptron.getErrorSignal() * lastInput2[i];
               hiddenLayerGradients[perceptronIndex][i] = grad;
            }
            perceptronIndex++;
        }

        for (int i = 0; i < expectedGradients.length; i++) {
            assertArrayEquals(expectedGradients[i], hiddenLayerGradients[i], 1e-4);
        }
    }

    @Test
    void multiLabelOutputLayerBackprop() {

        Perceptron out1 = new Perceptron(new double[]{0.5}, 0, "out1", Activations.linear());
        Perceptron out2 = new Perceptron(new double[]{0.5}, 0, "out2", Activations.linear());

        Net net = new Net.Builder()
                .addLayer(new Layer(new Perceptron[]{out1, out2}, Activations.linear()))
                .lossFn(Losses.MSE())
                .build();

        double[] input = {1.0};
        net.forward(input);

        double[] target = {1.0, 0.0};

        net.backpropagate(target);

        assertEquals(-1.0, out1.getErrorSignal(), 1e-4);
        assertEquals(1.0, out2.getErrorSignal(), 1e-4);
    }
}
