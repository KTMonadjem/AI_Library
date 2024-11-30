using Data.DataSet.Interface;
using Learning.Supervised.Training.Algorithm.Interface;
using Learning.Supervised.Training.LearningRate.Interface;
using Learning.Supervised.Training.LossFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Ann.Trainer;

public class BackPropagationWithGradientDescentTrainer : ITrainer
{
    private readonly Ann _ann;

    public BackPropagationWithGradientDescentTrainer(
        ILearningRate learningRate,
        ILossFunction lossFunction,
        IDataSet dataSet,
        Ann ann
    )
    {
        LearningRate = learningRate;
        LossFunction = lossFunction;
        DataSet = dataSet;
        _ann = ann;
    }

    public IDataSet DataSet { get; private init; }
    public ILearningRate LearningRate { get; private init; }
    public ILossFunction LossFunction { get; private init; }

    public void Train()
    {
        if (_ann is null)
            throw new NullReferenceException("Cannot train null Ann.");
        Vector<double> outputs;

        try
        {
            if (!_ann.HasBeenBuilt)
                _ann.Build();

            // This will run the Ann if it hasn't already been
            outputs = _ann.Outputs;
        }
        catch (Exception e)
        {
            throw new Exception("Error running the Ann during gradient descent: ", e);
        }
    }
}
