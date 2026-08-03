package msmarik;

import msmarik.activations.Activation;
import msmarik.initializers.Initializer;
import msmarik.losses.Loss;

public class Layer {
    private final int weightsNumber;
    private final int perceptronsNumber;
    private final Activation activationFn;
    private String name;
    private Perceptron[] perceptrons;

    public Layer(int weightsNumber, int perceptronsNumber, Activation activationFn) {
        this.weightsNumber = weightsNumber;
        this.perceptronsNumber = perceptronsNumber;
        this.activationFn = activationFn;

        validateLayerStructure();
    }

    /** Creates a layer whose perceptrons already contain explicit parameters. */
    public Layer(Perceptron[] perceptrons, Activation activationFn) {
        if (perceptrons == null || perceptrons.length == 0) {
            throw new IllegalArgumentException("Perceptrons cannot be null or empty");
        }

        if (perceptrons[0] == null) {
            throw new IllegalArgumentException("Perceptrons cannot contain null values");
        }

        this.perceptrons = perceptrons.clone();
        this.perceptronsNumber = perceptrons.length;
        this.weightsNumber = perceptrons[0].getWeights().length;
        this.activationFn = activationFn;

        validateLayerStructure();
        validatePerceptronShapes();
    }

    public void initializeLayer(Initializer weightInitializer) {
        if (isInitialized()) {
            return;
        }
        if (weightInitializer == null) {
            throw new IllegalStateException("Weight initializer is required for an uninitialized layer");
        }

        this.perceptrons = new Perceptron[perceptronsNumber];
        for (int i = 0; i < perceptrons.length; i++) {
            Perceptron p = new Perceptron(weightsNumber, perceptronsNumber, activationFn, weightInitializer);
            perceptrons[i] = p;
        }
    }

    public boolean isInitialized() {
        return perceptrons != null;
    }

    private void validatePerceptronShapes() {
        for (Perceptron perceptron : perceptrons) {
            if (perceptron == null) {
                throw new IllegalArgumentException("Perceptrons cannot contain null values");
            }
            if (perceptron.getWeights().length != weightsNumber) {
                throw new IllegalArgumentException("All perceptrons must have the same number of weights");
            }
        }
    }

    public Perceptron[] getPerceptrons() {
        return perceptrons;
    }

    public double[] forward(double[] input) {
        double[] output = new double[perceptrons.length];

        for (int i = 0; i < perceptrons.length; i++) {
            double perceptronOutput = perceptrons[i].computeWeightedSum(input);
            output[i] = perceptronOutput;
        }

        return output;
    }


    public void backpropagateOutputLayer(double[] target, Loss lossFn) {

        for (int i = 0; i < perceptrons.length; i++) {
            final Perceptron currentPerceptron = perceptrons[i];
            double predictedProbability = currentPerceptron.getResult();
            double error = lossFn.derivative(predictedProbability, target[i]);

            currentPerceptron.updateErrorSignal(error);
        }
    }

    public void backpropagateHiddenLayer(Layer nextLayer) {
        for (int i = 0; i < perceptrons.length; i++) {
            Perceptron currentPerceptron = perceptrons[i];
            double errorFromPreviousLayer = 0;

            for (int j = 0; j < nextLayer.perceptrons.length; j++) {
                Perceptron nextPerceptron = nextLayer.perceptrons[j];
                errorFromPreviousLayer += nextPerceptron.getWeights()[i] * nextPerceptron.getErrorSignal();
            }
            currentPerceptron.updateErrorSignal(errorFromPreviousLayer);
        }
    }

    public void updateWeightsAndBiases(double learningRate) {
        for (Perceptron perceptron: perceptrons) {
            perceptron.updateWeights(learningRate);
            perceptron.updateBias(learningRate);
        }
    }

    private void validateLayerStructure() {
        if (this.activationFn == null) throw new IllegalStateException("Activation function cannot be null.");
        if (this.perceptronsNumber < 1) throw new IllegalStateException("Layer must contain at least 1 perceptron");
        if (this.weightsNumber < 1)  throw new IllegalStateException("Perceptron must have at least 1 weight");
    }

    public void setLayerName(String name) {
        this.name = name;

        for (int i = 0; i < perceptrons.length; i++){
            perceptrons[i].setName(name + " - " + i);
        }
    }

    public int getWeightsNumber() {return this.weightsNumber;}
    public int getPerceptronsNumber() {return this.perceptronsNumber;}
    public Activation getActivationFn() {return this.activationFn;}
}
