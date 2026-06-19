package msmarik.initializers.functions;

import msmarik.initializers.Initializer;

import java.util.concurrent.ThreadLocalRandom;

public class He implements Initializer {
    @Override
    public double initializeWeight(int inputSize, int outputSize) {
        if (inputSize < 1) {
            throw new IllegalArgumentException("Input size must be at least 1");
        }

        double standardDeviation = Math.sqrt(2.0 / inputSize);
        return ThreadLocalRandom.current().nextGaussian() * standardDeviation;
    }
}
