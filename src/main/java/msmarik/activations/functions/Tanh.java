package msmarik.activations.functions;

import msmarik.activations.Activation;

import java.util.function.DoubleUnaryOperator;

public record Tanh() implements Activation {

    @Override
    public DoubleUnaryOperator standard() {
        return Math::tanh;
    }

    @Override
    public DoubleUnaryOperator derivative() {
        return x -> 1 - Math.pow(Math.tanh(x), 2);
    }
}
