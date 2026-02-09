package msmarik;

import msmarik.losses.Loss;
import msmarik.losses.Losses;

import java.util.ArrayList;

public class Net {
    private final ArrayList<Layer> layers;
    private final Loss lossFn;

    public Net(Loss lossFn) {
        layers = new ArrayList<>();
        this.lossFn = lossFn;
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

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

    public void updateWeights(double[] predictedProbabilities, double[] correctLabels) {
        double lastLayerGradient = this.lossFn.derivative(predictedProbabilities, correctLabels)[0];
        for (Layer layer : layers) {
            lastLayerGradient = layer.updateWeights(lastLayerGradient);
        }
    }
}
