using ANN.Structure.Layer;
using MathNet.Numerics.LinearAlgebra;

namespace ANN
{
    public class ANN
    {
        private bool _hasRun = false;
        private bool _inputsModified = true;
        private bool _hasBeenBuilt = false;
        private Vector<double> _outputs;


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

            _hasBeenBuilt = true;
            return this;
        }

        public void Run()
        {
            if (!_hasBeenBuilt)
            {
                throw new InvalidOperationException("ANN must be built before being run");
            }

            _outputs = Vector<double>.Build.Dense(Array.Empty<double>());

            _hasRun = true;
            _inputsModified = false;
        }
    }
}
