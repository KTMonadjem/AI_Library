using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.ANN.Structure
{
    public class Neuron
    {
        private bool _hasActivated = false;
        private double _output;

        public double Output
        {
            get
            {
                if (!_hasActivated)
                {
                    Activate();
                }
                return _output;
            }
        }
        public double Bias { get; }
        public Vector<double>? Inputs { get; private set; }
        public Vector<double> Weights { get; }
        public IActivationFunction Activator { get; }
        public List<Neuron>? Parents { get; private set; }

        private Neuron(Vector<double>? inputs, List<Neuron>? parents, Vector<double> weights, double bias, IActivationFunction activator)
        {
            Inputs = inputs;
            Parents = parents;
            Weights = weights;
            Bias = bias;
            Activator = activator;
        }

        public static Neuron Create(Vector<double> weights, double bias, IActivationFunction activator)
        {
            return new Neuron(null, null, weights, bias, activator);
        }

        public static Neuron CreateWithParents(List<Neuron> parents, Vector<double> weights, double bias, IActivationFunction activator)
        {
            if (parents.Count != weights.Count)
            {
                throw new ArgumentException("Neuron parents and weights must be the same length");
            }

            return new Neuron(null, parents, weights, bias, activator);
        }

        public static Neuron CreateWithInputs(Vector<double> inputs, Vector<double> weights, double bias, IActivationFunction activator)
        {
            if (inputs.Count != weights.Count)
            {
                throw new ArgumentException("Neuron inputs and weights must be the same length");
            }

            return new Neuron(inputs, null, weights, bias, activator);
        }

        public void SetParents(List<Neuron> parents)
        {
            if (Inputs != null)
            {
                throw new ArgumentException("Cannot set neuron parents when inputs are already set");
            }

            if (parents.Count != Weights.Count)
            {
                throw new ArgumentException("Neuron parents and weights must be the same length");
            }

            Parents = parents;
        }

        public void SetInputs(Vector<double> inputs)
        {
            if (Parents != null)
            {
                throw new ArgumentException("Cannot set neuron inputs when parents are already set");
            }

            if (inputs.Count != Weights.Count)
            {
                throw new ArgumentException("Neuron inputs and weights must be the same length");
            }

            Inputs = inputs;
        }


        public Vector<double> GetInputs()
        {
            if (Inputs is not null)
            {
                return Inputs;
            }
            else if (Parents is not null)
            {
                var inputs = new double[Parents.Count];
                for (var i = 0; i < Parents.Count; i++)
                {
                    inputs[i] = Parents[i].Output;
                }
                return Vector<double>.Build.Dense(inputs);
            }
            else
            {
                throw new ArgumentNullException();
            }
        }

        /// <summary>
        /// Modify each weight using gradient descent
        /// </summary>
        /// <param name="error"></param>
        /// <returns></returns>
        public Neuron? ModifyWeights(double error)
        {
            // Modify weights
            foreach (var weight in Weights)
            {

            }

            // Modify bias

            return null;
        }

        private double SumInputs()
        {
            return GetInputs().PointwiseMultiply(Weights).Sum() + Bias;
        }

        private void Activate()
        {
            _hasActivated = true;
            _output = Activator.Activate(SumInputs());
        }
    }
}
