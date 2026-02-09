package msmarik.losses.functions;

import msmarik.losses.Loss;

import java.util.function.DoubleUnaryOperator;

public record MSE() implements Loss {
    @Override
    public double standard(double[] predictedProbabilities, double[] correctLabels) {
        validateSizes(predictedProbabilities, correctLabels);

        double runningLoss = 0;
        for (int i = 0; i < predictedProbabilities.length; i++) {
            double loss = (correctLabels[i] - predictedProbabilities[i])
                    * (correctLabels[i] - predictedProbabilities[i]);
            runningLoss += loss;
        }

        return runningLoss / predictedProbabilities.length;
    }

    @Override
    public double[] derivative(double[] predictedProbabilities, double[] correctLabels) {
        validateSizes(predictedProbabilities, correctLabels);

        double[] resultGradients = new double[predictedProbabilities.length];
        for (int i = 0; i < predictedProbabilities.length; i++) {
            double resultGradient = ((double) 2 / predictedProbabilities.length)
                    * Math.signum(predictedProbabilities[i] - correctLabels[i]);
            resultGradients[i] = resultGradient;
        }
        return resultGradients;
    }
}
