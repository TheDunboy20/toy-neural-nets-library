package msmarik.losses.functions;

import msmarik.losses.Loss;

public record BCE() implements Loss {
    @Override
    public double standard(double[] predictedProbabilities, double[] correctLabels) {
        validateSizes(predictedProbabilities, correctLabels);

        double eps = 1e-15; // Prevent log(0)
        double runningLoss = 0;
        for (int i = 0; i < predictedProbabilities.length; i++) {
            double stableValue = Math.max(eps, Math.min(1-eps, predictedProbabilities[i]));
            double loss = - (correctLabels[i] * Math.log(stableValue)
                    + (1- correctLabels[i]) * Math.log(1 - stableValue));
            runningLoss += loss;
        }

        return runningLoss / predictedProbabilities.length;
    }

    @Override
    public double derivative(double predictedProbability, double correctLabel, int outputCount) {
        double eps = 1e-15;
        predictedProbability = Math.max(eps, Math.min(1 - eps, predictedProbability));

        return ((1 - correctLabel) / (1 - predictedProbability) - correctLabel / predictedProbability) / outputCount;
    }
}
