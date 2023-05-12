import { InferenceSession, Tensor, env } from 'onnxruntime-web';
import { HWEmojiPredictionResult, Points } from '../domain/model';
import log from '../log/logger';

export async function predictEmoji(points: Points): Promise<HWEmojiPredictionResult> {
    log("🔮 Starting prediction")
    log("   Loading model")
    const session = await InferenceSession.create("./hwemoji.onnx");
    const inputName = session.inputNames[0];
    log("   Input names " + session.inputNames);
    const outputName = session.outputNames[0];
    log("   Output names " + session.outputNames);
    log("   Model loaded")
    log("   Transforming input data")
    const inputData = transformPointsIntoData(points);
    log("   Input data ready")
    const dims = [1, inputData.length]
    const inputTensor = new Tensor('float32', inputData, dims);
    log(`   Evaluating HWEmoji model with tensor ${JSON.stringify(inputTensor.dims)}`)
    const feeds = { float_input: inputTensor };
    const outputMap = await session.run(feeds);
    const outputTensor = outputMap[outputName];
    log("   Prediction ready: " + JSON.stringify(outputMap))
    log("   Output tensor: " + JSON.stringify(outputTensor))
    // Get the predicted class from the output tensor
    const outputData = outputTensor.data;
    return [];
}

function transformPointsIntoData(points: Points): Float32Array {
    const result = Float32Array.from({ length: 100 * 100 }, () => 255.0);
    return result
}