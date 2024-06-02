import * as tf from "@tensorflow/tfjs";

//this function will take the prediction data and return a tensor of the mask
//shape of [width, height]
const ResizeMask = (predictionData, height, width) => {
  let mask = tf.tensor(predictionData, [256, 256]);
  mask = mask.expandDims(-1);
  const resizedMask = mask.resizeNearestNeighbor([height, width]);
  return resizedMask.squeeze();
};

//this function will return a tensor of binary mask based on the maskThreshold
//shape of [width, height]
const filterMaskPrediction = (predictionData, maskThreshold) => {
  return predictionData.greater(0.5).cast("float32");
};

const returnCentroid = (
  maskArray,
  maskShape,
  width,
  height,
  offsetX = 0,
  offsetY = 0
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
  return {
    centroidX: (sumX / count) * (width / maskShape[1]),
    centroidY: (sumY / count) * (height / maskShape[0]),
    centerX: width / 2 + offsetX,
    centerY: height / 2 + offsetY,
  };
};

function extractSquare(maskArray, center, size) {
  const [cx, cy] = center;
  const halfSize = Math.floor(size / 2);
  const startX = Math.max(cx - halfSize, 0);
  const startY = Math.max(cy - halfSize, 0);
  const endX = Math.min(cx + halfSize, maskArray[0].length);
  const endY = Math.min(cy + halfSize, maskArray.length);

  const square = [];
  for (let y = startY; y < endY; y++) {
    square.push(maskArray[y].slice(startX, endX));
  }
  return square;
}

// Function to detect shapes within the square using simple logic
function detectShapes(square, threshold = 1000) {
  const shapes = [];
  const visited = Array.from({ length: square.length }, () =>
    Array(square[0].length).fill(false)
  );

  function dfs(x, y) {
    const stack = [[x, y]];
    const shape = [];
    let count = 0;

    while (stack.length > 0) {
      const [cx, cy] = stack.pop();
      if (
        cx < 0 ||
        cy < 0 ||
        cx >= square[0].length ||
        cy >= square.length ||
        visited[cy][cx] ||
        square[cy][cx] === 0
      ) {
        continue;
      }

      visited[cy][cx] = true;
      count++;
      shape.push([cx, cy]);

      if (count > threshold) {
        return { count: count, shape: shape }; // Return early if threshold is exceeded
      }

      stack.push([cx + 1, cy]);
      stack.push([cx - 1, cy]);
      stack.push([cx, cy + 1]);
      stack.push([cx, cy - 1]);
    }

    return { count: count, shape: shape };
  }

  for (let y = 0; y < square.length; y++) {
    for (let x = 0; x < square[y].length; x++) {
      if (square[y][x] > 0 && !visited[y][x]) {
        const result = dfs(x, y);
        if (result.count > threshold) {
          return { exceededThreshold: true, shape: result.shape };
        }
        shapes.push(result);
      }
    }
  }

  return { exceededThreshold: false, shapes: shapes };
}

function checkShapesSize(shapes, threshold) {
  for (let shape of shapes) {
    if (shape >= threshold) {
      return true;
    }
  }
  return false;
}

export {
  ResizeMask,
  filterMaskPrediction,
  returnCentroid,
  extractSquare,
  detectShapes,
  checkShapesSize,
};
