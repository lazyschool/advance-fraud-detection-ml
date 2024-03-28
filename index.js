const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const modelPath = 'file://' + path.resolve('./randomforest/model.json');

async function predict(modelPath, inputTensor) {
    const model = await tf.loadLayersModel(modelPath);
    const inputTensor = tf.tensor([]).reshape([1, -1]); // Reshape as needed
    const prediction = model.predict(inputTensor);
    const outputData = await prediction.array(); // Use array() for asynchronous handling
    return outputData;
}

async function advanceFraudDetectionUsingML(input) {
    const randomforest = predict('file://' + path.resolve('./randomforest/model.json', input))
    const xgboost = predict('file://' + path.resolve('./xgboost/model.json', input))
    const gbboost = predict('file://' + path.resolve('./gbboost/model.json', input))
    const cnn = predict('file://' + path.resolve('./cnn/model.json', input))
    const ltsm = predict('file://' + path.resolve('./ltsm/model.json', input))
    const result = (randomforest.isFraud + xgboost.isFraud + gbboost.isFraud + cnn.isFraud + ltsm.isFraud) / 5
    result > 0.7 ? "This transaction may be a fraud" : "This transaction may not be a fraud"
}

module.exports = advanceFraudDetectionUsingML