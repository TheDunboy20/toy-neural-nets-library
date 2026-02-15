package msmarik.activations;

import msmarik.activations.functions.Linear;
import msmarik.activations.functions.Relu;
import msmarik.activations.functions.Sigmoid;
import msmarik.activations.functions.Tanh;

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

    public static Activation tanh() {
        return new Tanh();
    }
}
