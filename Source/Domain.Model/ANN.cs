using ANN.Interface;
using Learning.Supervised.ANN.Structure;
using MathNet.Numerics.LinearAlgebra;
using Training.Algorithm.Interface;

namespace Learning.Supervised.ANN
{
    public class ANN: IANN
    {
        private bool _hasRun = false;
        private bool _inputsModified = true;
        private bool _hasBeenBuilt = false;
        private Vector<double> _outputs;

        public ITrainer Trainer { get; private set; }
        public List<Layer> Layers { get; } = new List<Layer>();
        public Vector<double> Inputs { get; private set; } = Vector<double>.Build.Dense(Array.Empty<double>());
        public Vector<double> Outputs {
            get
            {
                if (!_hasRun)
                {
                    Run();
                }
                return _outputs;
            }
        }

        private ANN()
        {
        }

        private ANN(List<Layer> layers, Vector<double> inputs, ITrainer trainer)
        {
            SetInputs(inputs);
            AddLayers(layers);
            Trainer = trainer;
        }

        public static ANN Create()
        {
            return new ANN();
        }

        public static ANN Create(List<Layer> layers, Vector<double> inputs, ITrainer trainer)
        {
            return new ANN(layers, inputs, trainer);
        }

        public void AddLayer(Layer layer)
        {
            _hasBeenBuilt = false;
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

        public void SetTrainer(ITrainer trainer)
        {
            Trainer = trainer;
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

            _hasBeenBuilt = true;
            return this;
        }

        /// <summary>
        /// Run the inputs through this ANN into the output
        /// </summary>
        /// <exception cref="InvalidOperationException"></exception>
        public void Run()
        {
            if (!_hasBeenBuilt)
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

            _hasRun = true;
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
