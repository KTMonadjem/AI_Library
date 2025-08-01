﻿using MathNet.Numerics.LinearAlgebra;

namespace Common.Maths.ActivationFunction.Interface;

public interface IActivationFunction
{
    public (double Output, double Derivative) Activate(double input);

    public (Vector<double> Outputs, Vector<double> Derivatives) Activate(Vector<double> inputs);
}
