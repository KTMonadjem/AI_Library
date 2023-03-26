using ANN.Interface;
using Learning.Supervised.ANN.Structure;
using MathNet.Numerics.LinearAlgebra;
using Training.Algorithm.Interface;

namespace Learning.Supervised.ANN
{
    public class ANN: IANN
    {
        private bool _inputsModified = true;
        private Vector<double> _outputs;

        public List<Layer> Layers { get; } = new List<Layer>();
        public Vector<double> Inputs { get; private set; } = Vector<double>.Build.Dense(Array.Empty<double>());
        public bool HasRun { get; private set; }
        public bool HasBeenBuilt { get; private set; }
        public Vector<double> Outputs {
            get
            {
                if (!HasRun)
                {
                    Run();
                }
                return _outputs;
            }
        }

        private ANN()
        {
        }

        private ANN(List<Layer> layers, Vector<double> inputs)
        {
            SetInputs(inputs);
            AddLayers(layers);
        }

        public static ANN Create()
        {
            return new ANN();
        }

        public static ANN Create(List<Layer> layers, Vector<double> inputs)
        {
            return new ANN(layers, inputs);
        }

        public void AddLayer(Layer layer)
        {
            HasBeenBuilt = false;
            _inputsModified = true;
            Layers.Add(layer);
        }

        public void AddLayers(List<Layer> layers)
        {
            foreach (var layer in layers) 
            {
                AddLayer(layer);
            }
        }

        public void SetInputs(Vector<double> inputs)
        {
            Inputs = inputs;
        }

        /// <summary>
        /// Build the ANN graph from the weights and inputs
        /// </summary>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public ANN Build()
        {
            if (!Layers.Any())
            {
                throw new InvalidOperationException("ANN must have layers to build");
            }
            if (!Inputs.Any())
            {
                throw new InvalidOperationException("ANN must have inputs to build");
            }

            Layer? previous = null;
            foreach (var layer in Layers)
            {
                if (!layer.IsBuilt)
                {
                    layer.BuildWeights();
                }
                if (!layer.HasInputs)
                {
                    if (previous is null)
                    {
                        // If this is the first layer, use inputs instead of parents
                        layer.AddInputs(Inputs);
                    }
                    else
                    {
                        // Otherwise add parents to the current layer's inputs
                        layer.AddParentLayer(previous);
                    }
                }

                previous = layer;
            }

            HasBeenBuilt = true;
            return this;
        }

        /// <summary>
        /// Run the inputs through this ANN into the output
        /// </summary>
        /// <exception cref="InvalidOperationException"></exception>
        public void Run()
        {
            if (!HasBeenBuilt)
            {
                throw new InvalidOperationException("ANN must be built before being run");
            }

            // We only care about the last layers neurons.
            var finalLayerCount = Layers.Last().Neurons.Count;
            var outputs = new double[Layers.Last().Neurons.Count];
            for (var i = 0; i < finalLayerCount; i++)
            {
                // Fetching a neuron's output will fetch the parent's output too
                outputs[i] = Layers.Last().Neurons[i].Output;
            }

            _outputs = Vector<double>.Build.Dense(outputs);

            HasRun = true;
            _inputsModified = false;
        }

        /// <summary>
        /// Trains the ANN using the trainer
        /// </summary>
        public void Train()
        {

        }
    }
}
