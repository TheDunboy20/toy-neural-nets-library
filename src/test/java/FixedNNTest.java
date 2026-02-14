import msmarik.Layer;
import msmarik.Net;
import msmarik.Perceptron;
import msmarik.activations.Activations;
import msmarik.losses.Losses;

public interface FixedNNTest {
    default Net buildFixedNet() {
        Net net = new Net(Losses.MSE(), 1);

        // Layer 1
        Perceptron h1 = new Perceptron(Activations.linear(), new double[]{0.1, 0.2}, 0, "h1");
        Perceptron h2 = new Perceptron(Activations.linear(), new double[]{0.3, 0.4}, 0, "h2");
        Layer l1 = new Layer(new Perceptron[]{h1, h2});

        // Layer 2
        Perceptron h3 = new Perceptron(Activations.linear(), new double[]{0.5, 0.6}, 0, "h3");
        Perceptron h4 = new Perceptron(Activations.linear(), new double[]{0.7, 0.8}, 0, "h4");
        Layer l2 = new Layer(new Perceptron[]{h3, h4});

        // Output
        Perceptron h5 = new Perceptron(Activations.linear(), new double[]{0.9, 1.0}, 0, "h5");
        Layer l3 = new Layer(new Perceptron[]{h5});

        net.addLayers(l1, l2, l3);
        return net;
    }
}
