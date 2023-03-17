using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;
using SupervisedLearning.ANN.Neuron;

namespace ANN.Structure.Layer
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public Matrix<double> Weights { get; } 
        public IActivationFunction Activator { get; }

        private Layer(Matrix<double> weights, IActivationFunction activator)
        {
            Neurons = new List<Neuron>();
            Weights = weights;
            Activator = activator;
        }

        public static Layer Create(Matrix<double> weights, IActivationFunction activator) 
        { 
            if (weights.ColumnCount == 0 && weights.RowCount == 0)
            {
                throw new ArgumentException("Layer must be created with weights");
            }
            return new Layer(weights, activator); 
        }

        public Layer Build()
        {
            for (var i = 0; i < Weights.RowCount; i++)
            {
                var weights = Weights.Row(i);
                if (weights.Count == 0)
                {
                    continue;
                }

                var bias = weights[0];
                weights = weights.SubVector(1, weights.Count - 1);
                Neurons.Add(Neuron.Create(weights, bias, Activator.Activate));
            }

            return this;
        }
    }
}
