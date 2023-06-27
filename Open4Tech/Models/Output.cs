using Microsoft.ML.Data;

namespace Open4Tech.Models;

public class Output
{
    [VectorType(512)]
    public float[] input_ids { get; set; }

    [VectorType(512)]
    public float[] attention_mask { get; set; }
}
