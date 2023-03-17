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
        public bool IsBuilt { get; private set; } = false;
        public bool HasInputs { get; private set; } = false;

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

        /// <summary>
        /// Creates a layer with randomized weights. Additionally creates bias weights.
        /// </summary>
        /// <param name="numberOfNeurons"></param>
        /// <param name="numberOfWeights">The number of weights minus the bias weight</param>
        /// <param name="minWeight"></param>
        /// <param name="maxWeight"></param>
        /// <param name="activator"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
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

            // Add the bias neuron
            numberOfWeights++;

            var weights = new double[numberOfNeurons,numberOfWeights];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                for (int j = 0; j < numberOfWeights; j++)
                {
                    // Assign a randomly generated weight
                    weights[i,j] = _random.NextDouble() * (maxWeight - minWeight) + minWeight;
                }
            }

            return Create(Matrix<double>.Build.DenseOfArray(weights), activator);
        }

        /// <summary>
        /// Builds the weights for this layer from the weight matrix
        /// </summary>
        /// <returns></returns>
        public Layer BuildWeights()
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

            IsBuilt = true;
            return this;
        }

        /// <summary>
        /// Add inputs to the neurons in this layer
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public Layer AddInputs(Vector<double> inputs)
        {
            if (!inputs.Any())
            {
                throw new ArgumentException("Must have at least one input");
            }

            foreach (var neuron in Neurons)
            {
                neuron.SetInputs(inputs);
            }

            HasInputs = true;
            return this;
        }

        /// <summary>
        /// Link a set of parent neurons to the neurons in the layer.
        /// This does fully connected mapping.
        /// </summary>
        /// <param name="parents"></param>
        /// <returns></returns>
        public Layer AddParents(List<Neuron> parents)
        {
            if (!parents.Any())
            {
                throw new ArgumentException("Parents must be provided to add parents to this layer");
            }

            foreach (var neuron in Neurons)
            {
                neuron.SetParents(parents);
            }

            HasInputs = true;
            return this;
        }

        /// <summary>
        /// Adds another layer's neurons as the parents of the current layer
        /// </summary>
        /// <param name="layer"></param>
        /// <returns></returns>
        public Layer AddParentLayer(Layer layer)
        {
            return AddParents(layer.Neurons);
        }

        /// <summary>
        /// Deep copy this layer
        /// </summary>
        /// <returns></returns>
        public Layer Clone()
        {
            return Create(Weights.Clone(), Activator);
        }
    }
}
