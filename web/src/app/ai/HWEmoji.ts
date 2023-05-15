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
    let maxX = 0;
    let maxY = 0;
    let minX = 100000;
    let minY = 100000;
    const matrix: number[][] = Array.from({ length: 400 }, () => Array(400).fill(0));
    points.forEach(point => {   
        const x = Math.round(point.x);
        const y = Math.round(point.y);
        if (x > maxX) {
            maxX = x;
        }
        if (y > maxY) { 
            maxY = y;
        }
        if (x < minX) {
            minX = x;
        }
        if (y < minY) {
            minY = y;
        }
        matrix[y][x] = 1;
    });
    const canvas = document.createElement("canvas");
    canvas.width = matrix[0].length;
    canvas.height = matrix.length;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "#000000";
    context.fillRect(0, 0, canvas.width, canvas.height);
    for (let y = 0; y < matrix.length; y++) {
        for (let x = 0; x < matrix[0].length; x++) {
            context.fillStyle = "white";
            if (matrix[y][x] === 1) {
                context.fillRect(x, y, 3, 3);
            }
        }
    }
    log(`Data transformed into image ${canvas.toDataURL("image/png")}`)
    const croppedCanvas = document.createElement("canvas");
    croppedCanvas.width = maxX - minX + 1;
    croppedCanvas.height = maxY - minY + 1;
    const croppedContext = croppedCanvas.getContext("2d")!;
    croppedContext.drawImage(canvas, minX, minY, croppedCanvas.width, croppedCanvas.height, 0, 0, croppedCanvas.width, croppedCanvas.height);
    log(`Cropped data into image ${croppedCanvas.toDataURL("image/png")}`)
    const resultCanvas = document.createElement("canvas");
    resultCanvas.width = 100;
    resultCanvas.height = 100;
    const resultContext = resultCanvas.getContext("2d")!;
    resultContext.fillStyle = "#000000";
    resultContext.fillRect(0, 0, resultCanvas.width, resultCanvas.height);
    resultContext.drawImage(croppedCanvas, 0, 0, 100, 100);
    log(`Data resized into image ${resultCanvas.toDataURL("image/png")}`)
    return transformImageIntoFloat32Array(resultContext)
}

function transformImageIntoFloat32Array(context: CanvasRenderingContext2D): Float32Array {
    let resultArrayIndex = 0;
    const result = new Float32Array(100 * 100);
    const imageData = context.getImageData(0, 0, context.canvas.width, context.canvas.height);
    for (let y = 0; y < imageData.height; y++) {
        for (let x = 0; x < imageData.width; x++) {
            // Calculate the index of the pixel in the image data array
            const index = (y * imageData.width + x) * 4;
            // Call the callback function with the x, y, r, g, b, and a values of the pixel
            const r = imageData.data[index]
            const g =  imageData.data[index + 1]
            const b =  imageData.data[index + 2]
            const a = imageData.data[index + 3];
            const value = r + g+ b;
            if (value > 0) {
                result[resultArrayIndex] = 1;
            } else {
                result[resultArrayIndex] = 0;
            }
            resultArrayIndex += 1;
        }
    }
    return result
}