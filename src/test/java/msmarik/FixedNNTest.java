package msmarik;

import msmarik.activations.Activations;
import msmarik.losses.Losses;

public interface FixedNNTest {
    default Net buildFixedNet() {

        Perceptron[] layer1Perceptrons = new Perceptron[]{
                new Perceptron(new double[]{0.1, 0.2}, 0, "perceptron-1", Activations.linear()),
                new Perceptron(new double[]{0.3, 0.4}, 0, "perceptron-2", Activations.linear()),
        };

        Perceptron[] layer2Perceptrons = new Perceptron[]{
                new Perceptron(new double[]{0.5, 0.6}, 0, "perceptron-3", Activations.linear()),
                new Perceptron(new double[]{0.7, 0.8}, 0, "perceptron-4", Activations.linear()),
        };

        Perceptron[] layer3Perceptron = new Perceptron[]{
                new Perceptron(new double[]{0.9, 1.0}, 0, "perceptron-5", Activations.linear())
        };

        return new Net.Builder()
                .addLayer(new Layer(layer1Perceptrons, Activations.linear()))
                .addLayer(new Layer(layer2Perceptrons, Activations.linear()))
                .addLayer(new Layer(layer3Perceptron, Activations.linear()))
                .lossFn(Losses.MSE())
                .build();
    }
}
