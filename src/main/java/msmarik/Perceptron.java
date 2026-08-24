package msmarik;

import msmarik.activations.Activation;
import msmarik.initializers.Initializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Perceptron {
    private static final Logger log = LoggerFactory.getLogger(Perceptron.class);
    private double[] weights;
    private Double bias;
    private final Activation activationFn;
    private Initializer weightInitializer;
    private int weightsNumber;
    private int perceptronNumber;
    private double outputBeforeActivation;
    private double outputAfterActivation;
    private double errorSignal;
    private double[] lastInput;
    private String name;

    public Perceptron(int weightsNumber, int perceptronNumber, Activation activationFn,
                      Initializer weightInitializer) {
        this.weightsNumber = weightsNumber;
        this.perceptronNumber = perceptronNumber;
        this.activationFn = activationFn;
        this.weightInitializer = weightInitializer;

        initializeWeights();
    }

    public Perceptron(double[] weights, double bias, String name, Activation activationFn) {
        if (weights == null || weights.length == 0) {
            throw new IllegalArgumentException("Weights cannot be null or empty");
        }
        if (activationFn == null) {
            throw new IllegalArgumentException("Activation function cannot be null");
        }

        this.weights = weights.clone();
        this.bias = bias;
        this.name = name;
        this.activationFn = activationFn;
    }

    public double computeWeightedSum(double[] input) {
        if (this.weights.length != input.length) {
            System.out.println("Weights number: " + this.weightsNumber + " Input number: " + input.length);
            throw new IllegalArgumentException("Incorrect input size");
        }

        if (this.bias == null) {
            this.bias = initializeBias();
        }

        double result = 0;
        for (int i = 0; i < input.length; i++) {
            result += input[i] * weights[i];
        }
        result += bias;

        this.outputBeforeActivation = result;

        this.lastInput = input.clone();

        return result;
    }

    public void updateWeights(double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            log.debug("{} Weight[{}] update parameters: LR={}, ErrorSignal={}, LastInput={},", this.name, i,
                    learningRate,
                    errorSignal,
                    lastInput[i]);
            weights[i] -= learningRate * errorSignal * lastInput[i];
            log.debug("[{}] Updating weight[{}]. To: {} ", this.name, i, weights[i]);
        }
    }

    public void initializeWeights() {
        if (this.weights != null) {
            log.debug("Skipping weight initialization, weights already computed.");
            return;
        }
        this.weights = new double[weightsNumber];
        for (int i = 0; i < weightsNumber; i++) {
            weights[i] = weightInitializer.initializeWeight(weightsNumber, perceptronNumber);
        }
    }

    public double[] getWeights() {
        return weights.clone();
    }

    public double getErrorSignal() {
        return errorSignal;
    }

    public double[] getLastInput() {
        return lastInput.clone();
    }

    public void updateBias(double learningRate) {
        this.bias -= learningRate * errorSignal;
    }

    public double getBias() {
        return this.bias;
    }

    public double getOutputBeforeActivation() { return this.outputBeforeActivation; }

    public double getOutputAfterActivation() { return this.outputAfterActivation; }

    private double initializeBias() {
        return Math.random() - 0.5;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setOutputAfterActivation(double outputAfterActivation) { this.outputAfterActivation = outputAfterActivation; }

    public void setErrorSignal(double errorSignal) {this.errorSignal = errorSignal;}

}
