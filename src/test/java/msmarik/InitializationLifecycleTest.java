package msmarik;

import msmarik.activations.Activations;
import msmarik.losses.Losses;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class InitializationLifecycleTest {

    @Test
    void preinitializedLayerDoesNotRequireInitializer() {
        Perceptron perceptron = new Perceptron(
                new double[]{0.25, 0.75},
                0.1,
                "explicit",
                Activations.linear()
        );
        Layer layer = new Layer(new Perceptron[]{perceptron}, Activations.linear());

        Net net = new Net.Builder()
                .addLayer(layer)
                .lossFn(Losses.MSE())
                .build();

        assertSame(perceptron, net.getLayer(0).getPerceptrons()[0]);
        assertArrayEquals(new double[]{0.25, 0.75}, perceptron.getWeights());
        assertEquals(2, layer.getWeightsNumber());
        assertEquals(1, layer.getPerceptronsNumber());
    }

    @Test
    void uninitializedLayerRequiresInitializer() {
        Net.Builder builder = new Net.Builder()
                .addLayer(new Layer(2, 1, Activations.linear()))
                .lossFn(Losses.MSE());

        IllegalStateException exception = assertThrows(IllegalStateException.class, builder::build);

        assertEquals("Weight initializer is required for an uninitialized layer", exception.getMessage());
    }

    @Test
    void mixedNetworkOnlyInitializesLayersWithoutParameters() {
        Perceptron explicitPerceptron = new Perceptron(
                new double[]{0.5},
                0.0,
                "explicit",
                Activations.linear()
        );
        Layer explicitLayer = new Layer(new Perceptron[]{explicitPerceptron}, Activations.linear());
        Layer uninitializedLayer = new Layer(1, 1, Activations.linear());

        Net net = new Net.Builder()
                .addLayer(explicitLayer)
                .addLayer(uninitializedLayer)
                .weightInitializer((inputSize, outputSize) -> 0.75)
                .lossFn(Losses.MSE())
                .build();

        assertSame(explicitPerceptron, net.getLayer(0).getPerceptrons()[0]);
        assertArrayEquals(new double[]{0.5}, explicitPerceptron.getWeights());
        assertArrayEquals(new double[]{0.75}, net.getLayer(1).getPerceptrons()[0].getWeights());
    }
}
