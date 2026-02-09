package msmarik.losses;

import msmarik.losses.functions.BCE;
import msmarik.losses.functions.MAE;
import msmarik.losses.functions.MSE;

public class Losses {
    public static Loss BCE() {
        return new BCE();
    }

    public static Loss MAE() {
        return new MAE();
    }

    public static Loss MSE() {
        return new MSE();
    }
}
