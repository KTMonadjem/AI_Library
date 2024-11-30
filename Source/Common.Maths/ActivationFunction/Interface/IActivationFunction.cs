namespace Common.Maths.ActivationFunction.Interface;

public interface IActivationFunction
{
    public enum ActivationFunction
    {
        Binary,
        Linear,
        ReLu,
        LeakyReLu,
        ELu,
        Sigmoid,
        Tanh,
        Swish,
    }

    public double Delta { get; set; }
    public double Activate(double input);
}
