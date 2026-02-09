package msmarik;

import java.util.function.DoubleUnaryOperator;

public class Perceptron {
    private double[] weights;
    private double result;
    private double[] input;

    public double computeWeightedSum(double[] input, DoubleUnaryOperator activationFunction) {
        if (this.weights == null) {
            setPerceptronSize(input.length);
        }

        double result = 0;
        for (int i = 0; i < input.length; i++) {
            result += input[i] * weights[i];
            result = activationFunction.applyAsDouble(result);
        }

        this.input = input;
        this.result = result;

        return result;
    }

    public void updateWeights(double[] errorSignal, double learningRate) {
        // Probably one layer before the loss calculation?
        if (errorSignal.length != input.length) {
            for (int i = 0; i < input.length; i++) {
                double gradient = errorSignal[0] * input[i];
                weights[i] = weights[i] - (learningRate * gradient);
            }
        } else {

        }
    }

    private void setPerceptronSize(int inputLength) {
        this.weights = new double[inputLength];
        for (int i = 0; i < inputLength; i++) {
            this.weights[i] = 0.0001;
        }
    }

}
