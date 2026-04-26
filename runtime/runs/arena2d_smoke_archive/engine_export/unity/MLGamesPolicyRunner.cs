using System;
using System.Collections.Generic;

namespace MLGames.EngineExport
{
    public interface IMLGamesOnnxBackend
    {
        float[] Run(float[] observation);
    }

    public sealed class MLGamesPolicyRunner
    {
        public const int ObservationSize = 10;
        public const int ActionSize = 2;

        private readonly IMLGamesOnnxBackend _backend;

        public MLGamesPolicyRunner(IMLGamesOnnxBackend backend)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        }

        public float[] Predict(IReadOnlyList<float> observation)
        {
            if (observation == null || observation.Count != ObservationSize)
            {
                throw new ArgumentException($"Expected {ObservationSize} observation values.");
            }

            float[] input = new float[ObservationSize];
            for (int i = 0; i < ObservationSize; i++)
            {
                input[i] = observation[i];
            }

            float[] action = _backend.Run(input);
            if (action == null || action.Length != ActionSize)
            {
                throw new InvalidOperationException($"Backend must return {ActionSize} action values.");
            }

            return action;
        }
    }
}
