using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;
using SupervisedLearning.ANN.Neuron;

namespace ANN.Structure.Layer
{
    public class Layer
    {
        private static readonly Random _random = new Random();

        public List<Neuron> Neurons { get; }
        public Matrix<double> Weights { get; private set; } 
        public IActivationFunction Activator { get; }

        private Layer(Matrix<double> weights, IActivationFunction activator)
        {
            Neurons = new List<Neuron>();
            Weights = weights;
            Activator = activator;
        }

        public static Layer Create(Matrix<double> weights, IActivationFunction activator) 
        { 
            if (weights.ColumnCount <= 0 && weights.RowCount <= 0)
            {
                throw new ArgumentException("Layer must be created with weights");
            }
            return new Layer(weights, activator); 
        }

        public static Layer CreateWithRandomWeights(int numberOfNeurons, int numberOfWeights, double minWeight, double maxWeight, IActivationFunction activator)
        {
            if (numberOfNeurons <= 0)
            {
                throw new ArgumentException("Layer must be created with neurons");
            }
            if (numberOfWeights <= 0)
            {
                throw new ArgumentException("Layer must be created with weights");
            }

            if (minWeight > maxWeight)
            {
                throw new ArgumentException("Min weight must be less than max weight");
            }

            var weights = new double[numberOfNeurons,numberOfWeights];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                for (int j = 0; j < numberOfWeights; j++)
                {
                    weights[i,j] = _random.NextDouble() * (maxWeight - minWeight) + minWeight;
                }
            }

            return Create(Matrix<double>.Build.DenseOfArray(weights), activator);
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
