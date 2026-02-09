package msmarik.activations;

import msmarik.activations.functions.Linear;
import msmarik.activations.functions.Relu;
import msmarik.activations.functions.Sigmoid;

import java.util.function.DoubleUnaryOperator;

public class Activations {
    public static Activation relu() {
        return new Relu();
    }

    public static Activation linear() {
        return new Linear();
    }

    public static Activation sigmoid() {
        return new Sigmoid();
    }


    // TODO: Refactor this to use the same pattern as functions above
    public static DoubleUnaryOperator tanh() {
        return x -> 2 / ((1 + Math.exp(-2*x) -1));
    }

}
