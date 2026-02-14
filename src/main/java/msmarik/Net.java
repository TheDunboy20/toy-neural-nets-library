package msmarik;

import msmarik.losses.Loss;

import java.util.ArrayList;
import java.util.List;

public class Net {
    private final ArrayList<Layer> layers;
    private final Loss lossFn;
    private final double learningRate;

    private int layerIndex;

    public Net(Loss lossFn, double learningRate) {
        layers = new ArrayList<>();
        this.lossFn = lossFn;
        this.learningRate = learningRate;
    }

    public void addLayer(Layer layer) {
        layer.setLearningRate(learningRate);
        layer.setLayerName("Layer-" + layerIndex++);
        layers.add(layer);
    }

    public void addLayers(Layer... layersToAdd) {
        for (Layer layer: layersToAdd) {
            addLayer(layer);
        }
    }

    public List<Layer> getLayers() {
        return this.layers;
    }

    public Layer getLayer(int index) {
        return this.layers.get(index);
    };

    public double[] forward(double[] input) {
        double[] result = input.clone();

        for (Layer layer : layers) {
            result = layer.forward(result);
        }

        return result;
    }

    public double calculateLoss(double[] predictedProbabilities, double[] correctLabels) {
        return this.lossFn.standard(predictedProbabilities, correctLabels);
    }

    public void backpropagate(double correctLabels) {
        Layer outputLayer = layers.getLast();
        outputLayer.backpropagateOutputLayer(correctLabels, lossFn);

        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer current = layers.get(i);
            Layer next = layers.get(i + 1);

            current.backpropagateHiddenLayer(next);
        }
    }

    public void updateWeightsAndBiases(double learningRate) {
        for (Layer layer : layers) {
            layer.updateWeightsAndBiases();
        }
    }
}
