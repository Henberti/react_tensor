import * as tf from "@tensorflow/tfjs";

export const preprocessData = (dataTensor) => {
  return dataTensor
    .resizeNearestNeighbor([256, 256])
    .toFloat()
    .sub(tf.scalar(127.5))
    .div(tf.scalar(127.5))
    .expandDims(0);
};

export const reshapePrediction = (predictions) => {
  return tf.tensor2d(predictions, [256, 256]).expandDims(-1);
};

export const findMassCenter = (
  maskArray,
  videoHeight,
  videoWidth,
  maskShape
) => {
  let sumX = 0;
  let sumY = 0;
  let count = 0;
  for (let y = 0; y < maskArray.length; y++) {
    for (let x = 0; x < maskArray[y].length; x++) {
      if (maskArray[y][x] > 0) {
        sumX += x;
        sumY += y;
        count++;
      }
    }
  }

  const centroidX = (sumX / count) * (videoWidth / maskShape[1]);
  const centroidY = (sumY / count) * (videoHeight / maskShape[0]);

  return { centroidX, centroidY };
};
