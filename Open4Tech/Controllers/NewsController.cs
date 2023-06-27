using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using Open4Tech.Models;
using Open4Tech.Services;

namespace Open4Tech.Controllers;

[Route("api/[controller]/[action]")]
[ApiController]
public class NewsController : ControllerBase
{
    protected INewsService NewsService { get; set; }
    public NewsController(INewsService newsService)
    {
        NewsService = newsService;
    }

    [HttpGet]
    public async Task<IActionResult> GetAllNews()
    {
        var news = await NewsService.GetAllNews();
        return Ok(news);
    }

    [HttpGet]
    public async Task<IActionResult> GetNewsById(Guid id)
    {
        var news = await NewsService.GetNews(id);
        
        return (news is not null) ? Ok(news) : NotFound("News not found");
    }

    [HttpGet]
    public IActionResult TrainModel()
    {
        var mlContext = new MLContext();
        // Define the schema of the model's input data
        var dataSchema = new[]
        {
            new TextLoader.Column("input_ids", DataKind.Single, 0),
            new TextLoader.Column("attention_mask", DataKind.Single, 1)
        };

        // Load the ONNX model
        var pipeline = mlContext.Transforms.ApplyOnnxModel(
            modelFile: "model.onnx",
            outputColumnNames: new[] { "output" },
            inputColumnNames: new[] { "input_ids", "attention_mask" }
        );

        // Create an empty DataView to get the output schema
        var dataView = mlContext.Data.LoadFromEnumerable(new List<Input>());

        // Fit on empty data view
        var model = pipeline.Fit(dataView);

        // Create prediction engine
        var predEngine = mlContext.Model.CreatePredictionEngine<Input, Output>(model);

        // Prepare input data
        var input = new Input
        {
            input_ids = new float[512] /* fill with your input_ids data */,
            attention_mask = new float[512] /* fill with your attention_mask data */
        };

        // Make prediction
        var result = predEngine.Predict(input);


        return Ok();
    }
}
