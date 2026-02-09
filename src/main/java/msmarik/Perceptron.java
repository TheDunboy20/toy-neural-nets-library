package msmarik;

import java.util.function.DoubleUnaryOperator;

public class Perceptron {
    private double[] weights;

    public double computeWeightedSum(double[] input, DoubleUnaryOperator activationFunction) {
        if (this.weights == null) {
            setPerceptronSize(input.length);
        }

        double result = 0;
        for (int i = 0; i < input.length; i++) {
            result += input[i] * weights[i];
            result = activationFunction.applyAsDouble(result);
        }

        return result;
    }

    private void setPerceptronSize(int inputLength) {
        this.weights = new double[inputLength];
        for (int i = 0; i < inputLength; i++) {
            this.weights[i] = 0.0001;
        }
    }
}
