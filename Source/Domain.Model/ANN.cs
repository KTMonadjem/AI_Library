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


        public List<Layer> Layers { get; }
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
            Layers = new List<Layer>();
        }

        private ANN(List<Layer> layers, Vector<double> inputs)
        {
            Layers = new List<Layer>();

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
            _hasBeenBuilt = true;

            return this;
        }

        public void Run()
        {
            if (!_hasBeenBuilt)
            {
                throw new InvalidOperationException("ANN must be built before being run");
            }

            _hasRun = true;
            _inputsModified = false;
            _outputs = Vector<double>.Build.Dense(Array.Empty<double>());
        }
    }
}
