package msmarik;

import msmarik.activations.Activation;

public class Perceptron {
    private double[] weights;
    private double bias;
    private final Activation activationFn;
    private double result;
    private double outputBeforeActivation;
    private double errorSignal;
    private double[] lastInput;

    public Perceptron(Activation activationFn) {
        this.activationFn = activationFn;
        this.bias = initializeBias();
    }

    public double computeWeightedSum(double[] input) {
        if (this.weights == null) {
            setPerceptronSize(input.length);
        }

        double result = 0;
        for (int i = 0; i < input.length; i++) {
            result += input[i] * weights[i];
        }
        result += bias;

        this.outputBeforeActivation = result;
        result = activationFn.standard().applyAsDouble(result);
        this.lastInput = input;
        this.result = result;

        return result;
    }

    private void setPerceptronSize(int inputLength) {
        this.weights = new double[inputLength];
        for (int i = 0; i < inputLength; i++) {
            this.weights[i] = 0.0001;
        }
    }

    public void updateErrorSignal(double errorSignalFromPreviousLayer) {
        this.errorSignal = errorSignalFromPreviousLayer
                         * activationFn.derivative().applyAsDouble(outputBeforeActivation);
    }

    public void updateWeights(double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * errorSignal * lastInput[i];
        }
    }

    public double[] getWeights() {
        return weights;
    }

    public double getErrorSignal() {
        return errorSignal;
    }

    public double getResult() {
        return result;
    }

    public void updateBias(double learningRate) {
        this.bias -= learningRate * errorSignal;
    }

    private double initializeBias() {
        return Math.random() - 0.5;
    }


}
