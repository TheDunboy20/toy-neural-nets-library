package msmarik.losses;

public interface Loss {
    double standard(double[] predictedProbabilities, double[] correctLabels);
    double derivative(double predictedProbability, double correctLabel, int outputCount);

    default void validateSizes(double[] predictedProbabilities, double[] correctLabels) {
        if (correctLabels.length != predictedProbabilities.length) {
            throw new IllegalArgumentException("Predicted size + " + predictedProbabilities.length + " != label size + " + correctLabels.length);
        }
    }
}
