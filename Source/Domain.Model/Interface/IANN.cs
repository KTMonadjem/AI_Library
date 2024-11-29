using Learning.Supervised.Training.Algorithm.Interface;

namespace ANN.Interface;

public interface IANN : ILearner
{
    public void Run();
}