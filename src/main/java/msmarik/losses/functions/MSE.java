package msmarik.losses.functions;

import msmarik.losses.Loss;

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
    public double derivative(double predictedProbability, double correctLabel) {
        return 2  * (predictedProbability - correctLabel);
    }
}
