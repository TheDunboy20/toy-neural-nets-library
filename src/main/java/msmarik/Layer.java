package msmarik;

import msmarik.activations.Activation;
import msmarik.losses.Loss;

public class Layer {
    private final Perceptron[] perceptrons;
    private double learningRate;
    private String name;

    public Layer(int outputFeatureSize, Activation activationFn) {
        this.perceptrons = new Perceptron[outputFeatureSize];

        for (int i = 0; i < outputFeatureSize; i++) {
            perceptrons[i] = new Perceptron(activationFn, name + "-" + i);
        }
    }

    /*
    * For testing purposes only
    * */
    public Layer(Perceptron[] perceptrons) {
        this.perceptrons = perceptrons;
    }

    public Perceptron[] getPerceptrons() {
        return perceptrons;
    }

    public double[] forward(double[] input) {
        double[] output = new double[perceptrons.length];

        for (int i = 0; i < perceptrons.length; i++) {
            double perceptronOutput = perceptrons[i].computeWeightedSum(input);
            output[i] = perceptronOutput;
        }

        return output;
    }

    public void setLearningRate(double lr) {
        this.learningRate = lr;
    }

    public void backpropagateOutputLayer(double target, Loss lossFn) {
        for (Perceptron currentPerceptron : perceptrons) {
            double predictedProbability = currentPerceptron.getResult();
            double error = lossFn.derivative(predictedProbability, target);

            currentPerceptron.updateErrorSignal(error);
        }
    }

    public void backpropagateHiddenLayer(Layer nextLayer) {
        for (int i = 0; i < perceptrons.length; i++) {
            Perceptron currentPerceptron = perceptrons[i];
            double errorFromPreviousLayer = 0;

            for (int j = 0; j < nextLayer.perceptrons.length; j++) {
                Perceptron nextPerceptron = nextLayer.perceptrons[j];
                errorFromPreviousLayer += nextPerceptron.getWeights()[i] * nextPerceptron.getErrorSignal();
            }
            currentPerceptron.updateErrorSignal(errorFromPreviousLayer);
        }
    }

    public void updateWeightsAndBiases() {
        for (Perceptron perceptron: perceptrons) {
            perceptron.updateWeights(this.learningRate);
            perceptron.updateBias(this.learningRate);
        }
    }

    public void setLayerName(String name) {
        this.name = name;
    }
}
