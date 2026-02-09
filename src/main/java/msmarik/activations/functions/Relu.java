package msmarik.activations.functions;

import msmarik.activations.Activation;

import java.util.function.DoubleUnaryOperator;

public record Relu() implements Activation {

    @Override
    public DoubleUnaryOperator standard() {
        return x -> Math.max(0, x);
    }

    @Override
    public DoubleUnaryOperator derivative() {
        return x -> (x > 0 ? 1 : 0);
    }
}
