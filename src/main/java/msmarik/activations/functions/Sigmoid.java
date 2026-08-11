package msmarik.activations.functions;

import msmarik.activations.Activation;
import msmarik.activations.DoubleArrayUnaryOperator;

public record Sigmoid() implements Activation {
    @Override
    public DoubleArrayUnaryOperator standard() {
        return elementwise(x -> (1 / (1 + Math.exp(-x))));
    }

    @Override
    public DoubleArrayUnaryOperator derivative() {
        return elementwise(x -> (Math.exp(-x) / Math.pow((1 + Math.exp(-x)), 2)));
    }
}
