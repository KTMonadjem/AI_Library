namespace Common.Maths.ActivationFunction.Interface
{
    public interface IActivationFunction: IActivationDerivative
    {
        public double Delta { get; set; }
        public abstract double Activate(double input);
    }
}
