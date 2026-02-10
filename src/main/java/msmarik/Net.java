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
        Layer previousLayer = layers.isEmpty() ? null : layers.getLast();
        layer.setPreviousLayer(previousLayer);
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

//    public void updateWeights(double[] predictedProbabilities, double[] correctLabels) {
//        double[] errorSignal = this.lossFn.derivative(predictedProbabilities, correctLabels);
//        for (Layer layer : layers) {
//            errorSignal = layer.updateWeights(errorSignal);
//        }
//    }
}
