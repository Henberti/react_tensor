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
  return tf.tidy(() => {
    const coloredTensor = tf.zeros([height, width, 3], "float32");
    const coloredTensorBuffer = coloredTensor.bufferSync();

    shapes
      .sort((a, b) => a.length > b.length)
      .forEach((shapeInfo, index) => {
        const color = Colors[index % Colors.length]; // Random color for each shape

        shapeInfo.shape.forEach(([x, y]) => {
          if (x < width && y < height) {
            const offsetX = x + centeredX - 50;
            const offsetY = y + centeredY - 50;
            if (offsetX >= 0 && offsetY >= 0 && offsetX < width && offsetY < height) {
              coloredTensorBuffer.set(color[0], offsetY, offsetX, 0);
              coloredTensorBuffer.set(color[1], offsetY, offsetX, 1);
              coloredTensorBuffer.set(color[2], offsetY, offsetX, 2);
            }
          }
        });
      });

    return coloredTensorBuffer.toTensor().mul(255);
  });
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
    return tensorA.mul(alpha).add(tensorB.mul(1 - alpha));
  });
};

const addAlphaChannel = (rgbTensor, alphaChannel, width, height) => {
  return tf.tidy(() => {
    const alphaChannelReshaped = alphaChannel.reshape([height, width, 1]);
    return tf.concat([rgbTensor, alphaChannelReshaped], -1);
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
  return tf.tidy(() => {
    const alphaChannel = mask2d.mul(127).add(128);
    const Xmin = Math.max(xStart - detectionBoundaryBox, 0);
    const Ymin = Math.max(yStart - detectionBoundaryBox, 0);

    if (isValidCenter) {
      drawFilledSquare(ctx, xStart, yStart, 20, "green");
    }
    drawBoundingBox(ctx, Xmin, Ymin, 200, isValidCenter ? "green" : "yellow");
    drawFilledSquare(ctx, Math.round(width / 2), Math.round(height / 2), 20, "red");

    let videoTensor = tf.browser.fromPixels(video);
    if (coloredTensor) {
      videoTensor = blendTensors(videoTensor, coloredTensor, 0.5);
    }

    return addAlphaChannel(videoTensor, alphaChannel, width, height);
  });
};

export {
  drawFilledSquare,
  drawMask,
  drawBoundingBox,
  blendTensors,
  addAlphaChannel,
  createColoredTensor,
};
