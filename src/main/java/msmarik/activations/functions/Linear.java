package msmarik.activations.functions;

import msmarik.activations.Activation;

import java.util.function.DoubleUnaryOperator;

public record Linear() implements Activation {
    @Override
    public DoubleUnaryOperator standard() {
        return DoubleUnaryOperator.identity();
    }

    @Override
    public DoubleUnaryOperator derivative() {
        return x -> 1;
    }
}
