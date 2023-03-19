namespace Common.Maths.ActivationFunction.Interface
{
    public interface IActivationFunction: IActivationDerivative
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
            Swish
        }

        public abstract double Delta { get; set; }
        public abstract double Activate(double input);
    }
}
