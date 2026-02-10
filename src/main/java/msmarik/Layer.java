package msmarik;

import msmarik.activations.Activation;

public class Layer {

    private final Perceptron[] perceptrons;
    private final Activation activationFunction;

    private Layer previousLayer;
    private double[] layerInput;

    public Layer(int outputFeatureSize, Activation activationFunction) {
        this.perceptrons = new Perceptron[outputFeatureSize];
        this.activationFunction = activationFunction;

        for (int i = 0; i < outputFeatureSize; i++) {
            perceptrons[i] = new Perceptron();
        }
    }

    public double[] forward(double[] input) {
        this.layerInput = input.clone();

        double[] output = new double[perceptrons.length];

        for (int i = 0; i < perceptrons.length; i++) {
            double perceptronOutput = perceptrons[i].computeWeightedSum(input, activationFunction.standard());
            output[i] = perceptronOutput;
        }

        return output;
    }

//    public double[] updateWeights(double[] errorSignal) {
//        for (int i = 0; i < perceptrons.length; i++) {
//            perceptrons.updateWeights(errorSignal);
//        }
//    }

    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    private Activation getActivationFunction() {
        return activationFunction;
    }
}
