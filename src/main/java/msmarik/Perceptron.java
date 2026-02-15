package msmarik;

import msmarik.activations.Activation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class Perceptron {
    private static final Logger log = LoggerFactory.getLogger(Perceptron.class);
    private double[] weights;
    private Double bias;
    private final Activation activationFn;
    private double result;
    private double outputBeforeActivation;
    private double errorSignal;
    private double[] lastInput;
    private final String name;

    public Perceptron(Activation activationFn, String name) {
        this.activationFn = activationFn;
        this.name = name;
    }

    public Perceptron(Activation activationFn, double[] weights, double bias, String name) {
        this.activationFn = activationFn;
        this.weights = weights;
        this.bias = bias;
        this.name = name;
    }

    public double computeWeightedSum(double[] input) {
        if (this.weights == null) {
            setPerceptronSize(input.length);
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
        result = activationFn.standard().applyAsDouble(result);

        log.debug("[{}] Output after activation: {}", this.name, result);

        this.lastInput = input;
        this.result = result;

        return result;
    }

    private void setPerceptronSize(int inputLength) {
        this.weights = new double[inputLength];
        for (int i = 0; i < inputLength; i++) {
            this.weights[i] = 0.0001;
        }
    }

    public void updateErrorSignal(double errorSignalFromPreviousLayer) {
        log.info("[{}] Error signal passed from previous layer: {}", this.name, errorSignalFromPreviousLayer);
        this.errorSignal = errorSignalFromPreviousLayer
                         * activationFn.derivative().applyAsDouble(outputBeforeActivation);
        log.info("[{}] Error signal updated to: {}", this.name, errorSignal);
    }

    public void updateWeights(double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            log.info("{} Weight[{}] update parameters: LR={}, ErrorSignal={}, LastInput={},", this.name, i,
                    learningRate,
                    errorSignal,
                    lastInput[i]);
            weights[i] -= learningRate * errorSignal * lastInput[i];
            log.info("[{}] Updating weight[{}]. To: {} ", this.name, i, weights[i]);
        }
    }

    public double[] getWeights() {
        return weights;
    }

    public double getErrorSignal() {
        return errorSignal;
    }

    public double getResult() {
        return result;
    }

    public double[] getLastInput() {
        return lastInput;
    }

    public void updateBias(double learningRate) {
        this.bias -= learningRate * errorSignal;
    }

    public double getBias() {
        return this.bias;
    }

    private double initializeBias() {
        return Math.random() - 0.5;
    }


}
