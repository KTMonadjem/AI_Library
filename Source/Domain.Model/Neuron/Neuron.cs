using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Domain.Model.Neuron
{
    public class Neuron
    {
        public double Output { get; }
        public Matrix<double> Inputs { get; }
        private Func<double, double> Activator { get; }

        private Neuron(Matrix<double> inputs, Func<double, double> activator)
        {
            Inputs = inputs;
            Activator = activator;
        }

        public static Neuron Create(Matrix<double> inputs, Func<double, double> activator)
        {
            return new Neuron(inputs, activator);
        }

        public double Activate()
        {
            return Activator(SumInputs());
        }

        private double SumInputs()
        {
            return 0.0;
        }
    }
}
