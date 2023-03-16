using MathNet.Numerics.LinearAlgebra;

namespace SupervisedLearning.ANN.Neuron
{
    public class Neuron
    {
        public double Output { get; private set; }
        public double Bias { get; }
        public Vector<double> Inputs { get; }
        public Vector<double> Weights { get; }
        public Func<double, double> Activator { get; }

        private Neuron(Vector<double> inputs, Vector<double> weights, double bias, Func<double, double> activator)
        {
            Inputs = inputs;
            Weights = weights;
            Bias = bias;
            Activator = activator;
        }

        public static Neuron Create(Vector<double> inputs, Vector<double> weights, double bias, Func<double, double> activator)
        {
            return new Neuron(inputs, weights, bias, activator);
        }

        public void Activate()
        {
            Output = Activator(SumInputs());
        }

        private double SumInputs()
        {
            return Inputs.PointwiseMultiply(Weights).Sum() + Bias;
        }
    }
}
