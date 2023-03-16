using MathNet.Numerics.LinearAlgebra;

namespace SupervisedLearning.ANN.Neuron
{
    public class Neuron
    {
        private bool _hasActivated = false;
        private double _output;

        public double Output { 
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
        public Vector<double>? Inputs { get; }
        public Vector<double> Weights { get; }
        public Func<double, double> Activator { get; }
        public List<Neuron>? Parents { get; }

        private Neuron(Vector<double>? inputs, List<Neuron>? parents, Vector<double> weights, double bias, Func<double, double> activator)
        {
            Inputs = inputs;
            Parents = parents;
            Weights = weights;
            Bias = bias;
            Activator = activator;
        }

        public static Neuron CreateWithParents(List<Neuron>parents, Vector<double> weights, double bias, Func<double, double> activator)
        {
            return new Neuron(null, parents, weights, bias, activator);
        }

        public static Neuron CreateWithInputs(Vector<double> inputs, Vector<double> weights, double bias, Func<double, double> activator)
        {
            return new Neuron(inputs, null, weights, bias, activator);
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

        private double SumInputs()
        {
            return GetInputs().PointwiseMultiply(Weights).Sum() + Bias;
        }

        private void Activate()
        {
            _hasActivated = true;
            _output = Activator(SumInputs());
        }
    }
}
