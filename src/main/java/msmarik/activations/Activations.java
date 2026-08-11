package msmarik.activations;

import msmarik.activations.functions.*;

public class Activations {
    public static Activation relu() {
        return new Relu();
    }

    public static Activation linear() {
        return new Linear();
    }

    public static Activation softmax() {return new Softmax();}

    public static Activation sigmoid() {
        return new Sigmoid();
    }

    public static Activation tanh() {
        return new Tanh();
    }
}
