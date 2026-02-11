package msmarik;

import msmarik.activations.Activation;
import msmarik.losses.Loss;

public class Layer {
    private final Perceptron[] perceptrons;
    private double learningRate;

    public Layer(int outputFeatureSize, Activation activationFn) {
        this.perceptrons = new Perceptron[outputFeatureSize];

        for (int i = 0; i < outputFeatureSize; i++) {
            perceptrons[i] = new Perceptron(activationFn);
        }
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

    public void backpropagateOutputLayer(double[] target, Loss lossFn) {
        for (int i = 0; i < perceptrons.length; i++) {
            Perceptron currentPerceptron = perceptrons[i];
            // Artificially wrapping in an array, to support batch functionality in the future.
            double[] targets = {target[i]};
            double[] predictedProbabilities = {currentPerceptron.getResult()};

            double[] error = lossFn.derivative(predictedProbabilities, targets);
            currentPerceptron.updateErrorSignal(error[0]);
            currentPerceptron.updateWeights(this.learningRate);
            currentPerceptron.updateBias(this.learningRate);
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
            currentPerceptron.updateWeights(this.learningRate);
            currentPerceptron.updateBias(this.learningRate);
        }
    }
}
