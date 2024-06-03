import * as tf from "@tensorflow/tfjs";

const Colors = [
  [0.9, 0.1, 0.1], // Soft Red
  [0.1, 0.9, 0.1], // Soft Green
  [0.1, 0.1, 0.9], // Soft Blue
  [0.9, 0.9, 0.1], // Soft Yellow
  [0.1, 0.9, 0.9], // Soft Cyan
  [0.9, 0.1, 0.9], // Soft Magenta
];

function createColoredTensor(shapes, height, width, centeredX, centeredY) {
  const coloredTensor = tf.zeros([height, width, 3], "float32");
  const coloredTensorBuffer = coloredTensor.bufferSync();

  shapes
    .sort((a, b) => a.length > b.length)
    .forEach((shapeInfo, index) => {
      const color = Colors[index % Colors.length]; // Random color for each shape

      shapeInfo.shape.forEach(([x, y]) => {
        if (x < width && y < height) {
          coloredTensorBuffer.set(
            color[0],
            y + centeredY - 50,
            x + centeredX - 50,
            0
          );
          coloredTensorBuffer.set(
            color[1],
            y + centeredY - 50,
            x + centeredX - 50,
            1
          );
          coloredTensorBuffer.set(
            color[2],
            y + centeredY - 50,
            x + centeredX - 50,
            2
          );
        }
      });
    });

  return coloredTensorBuffer.toTensor().mul(255);
}


const drawFilledSquare = (ctx, x, y, size, color) => {
  ctx.fillStyle = color;
  ctx.fillRect(x, y, size, size);
};

const drawBoundingBox = (ctx, x, y, size, color) => {
  ctx.strokeStyle = color;
  ctx.strokeRect(x, y, size, size);
};

const blendTensors = (tensorA, tensorB, alpha) => {
  return tf.tidy(() => {
    const blendedTensor = tensorA.mul(alpha).add(tensorB.mul(1 - alpha));
    return blendedTensor;
  });
};
const addAlphaChannel = (rgbTensor, alphaChannel) => {
  return tf.tidy(() => {
    // Ensure the alpha channel has shape [480, 640, 1]
    const alphaChannelReshaped = alphaChannel.reshape([480, 640, 1]);

    // Concatenate along the last axis to form the RGBA tensor
    const rgbaTensor = tf.concat([rgbTensor, alphaChannelReshaped], -1);

    return rgbaTensor;
  });
};

const drawMask = (
  ctx,
  mask2d,
  isValidCenter,
  video,
  coloredTensor,
  xStart,
  yStart,
  width,
  height,
  detectionBoundaryBox = 100
) => {
  const alphaChannel = mask2d.mul(127).add(128);
  const Xmin = Math.max(xStart - detectionBoundaryBox, 0);


  const Ymin = Math.max(yStart - detectionBoundaryBox, 0);
  if (isValidCenter) {
    drawFilledSquare(ctx, xStart, yStart, 20, "green");
  }
  drawBoundingBox(ctx, Xmin, Ymin, 200, isValidCenter?"green": 'yellow');
  drawFilledSquare(
    ctx,
    Math.round(width / 2),
    Math.round(height / 2),
    20,
    "red"
  );
  let t2 = tf.browser.fromPixels(video);
  if (coloredTensor) {
    t2 = blendTensors(t2, coloredTensor, 0.5);
  }
  const ttt2 = addAlphaChannel(t2, alphaChannel);
  
  return ttt2;
};

export {
  drawFilledSquare,
  drawMask,
  drawBoundingBox,
  blendTensors,
  addAlphaChannel,
  createColoredTensor,
};
