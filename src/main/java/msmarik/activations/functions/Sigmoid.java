package msmarik.activations.functions;

import msmarik.activations.Activation;

import java.util.function.DoubleUnaryOperator;

public record Sigmoid() implements Activation {
    @Override
    public DoubleUnaryOperator standard() {
        return x -> (1 / (1 + Math.exp(-x)));
    }

    @Override
    public DoubleUnaryOperator derivative() {
        return x -> (Math.exp(-x) / Math.pow((1 + Math.exp(-x)), 2));
    }
}
