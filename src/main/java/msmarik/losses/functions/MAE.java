package msmarik.losses.functions;

import msmarik.losses.Loss;

public record MAE() implements Loss {
    @Override
    public double standard(double[] predictedProbabilities, double[] correctLabels) {
        validateSizes(predictedProbabilities, correctLabels);

        double runningLoss = 0;
        for (int i = 0; i < predictedProbabilities.length; i++) {
            double loss = Math.abs(correctLabels[i] - predictedProbabilities[i]);
            runningLoss += loss;
        }

        return runningLoss / predictedProbabilities.length;
    }

    @Override
    public double derivative(double predictedProbability, double correctLabel) {
        double diff = predictedProbability - correctLabel;

        if (diff > 0) return 1;
        if (diff < 0) return -1;
        return 0;
    }
}
