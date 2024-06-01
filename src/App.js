import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import DetectModel from "./DetectModel";
import tts from "./Components/Tts";
//this is the model that we will use to predict the mask hen
const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef();
  const wasRendered = useRef(false);
  const canvas2Ref = useRef();
  const [model, setModel] = useState(null);
  const lastTtsCall = useRef(Date.now());
  let voices = window.speechSynthesis.getVoices();
  const voice = voices[1];
  const Colors = [
    [0.9, 0.1, 0.1],  // Soft Red
    [0.1, 0.9, 0.1],  // Soft Green
    [0.1, 0.1, 0.9],  // Soft Blue
    [0.9, 0.9, 0.1],  // Soft Yellow
    [0.1, 0.9, 0.9],  // Soft Cyan
    [0.9, 0.1, 0.9]   // Soft Magenta
  ];

  const debounceTts = (message) => {
    const now = Date.now();

    if (now - lastTtsCall.current > 3000) {
      tts(message, voice);
      lastTtsCall.current = now;
    }
  };

  useEffect(() => {
    tf.loadLayersModel(process.env.PUBLIC_URL + "models/jsconv2/model.json")
      .then((loadedModel) => {
        console.log("Model loaded successfully:", loadedModel);
        setModel(loadedModel);
      })
      .catch((error) => {
        console.error("Error loading model:", error);
      });
  }, []);

  useEffect(() => {
    if (!model || !videoRef.current) return;
    if (wasRendered.current) return;
    wasRendered.current = true;

    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
        videoRef.current.play().then(() => {
          captureAndPredict();
        });
      })
      .catch((error) => {
        console.error("Error accessing webcam:", error);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  const captureAndPredict = async () => {
    if (!videoRef.current || !model) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    const canvas2 = canvas2Ref.current;

    const drawFrame = () => {
      tf.tidy(() => {
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;
        canvas.width = videoWidth;
        canvas.height = videoHeight;

        context.drawImage(video, 0, 0, videoWidth, videoHeight);

        const tensCanvas = tf.browser.fromPixels(canvas);

        // Preprocess the frame
        const tensor = tensCanvas
          .resizeNearestNeighbor([256, 256])
          .toFloat()
          .sub(tf.scalar(127.5))
          .div(tf.scalar(127.5))
          .expandDims(0);

        model
          .predict(tensor)
          .data()
          .then((predictionData) => {
            tf.tidy(() => {
              // Assuming predictionData is a flat array; convert to a 2D tensor (256x256) as before
              let maskTensor = tf.tensor2d(predictionData, [256, 256]);

              // Expand dimensions to make it [256, 256, 1], which is necessary for resizeNearestNeighbor
              maskTensor = maskTensor.expandDims(-1);

              // Now resize the mask to match the video frame dimensions
              const resizedMask = maskTensor.resizeNearestNeighbor([
                videoHeight,
                videoWidth,
              ]);

              // Remove the singleton dimension after resizing so we can use it for masking
              let mask2d = resizedMask.squeeze();
              mask2d = mask2d.greater(0.5).cast("float32");

              const maskArray = mask2d.arraySync();
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

              const centroidX = (sumX / count) * (videoWidth / mask2d.shape[1]);
              const centroidY =
                (sumY / count) * (videoHeight / mask2d.shape[0]);

              const centerX = videoWidth / 2;
              const centerY = videoHeight / 2;
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

              function createColoredTensor(shapes, height, width, centeredX, centeredY) {
                const coloredTensor = tf.zeros([height, width, 3], "float32");
                const coloredTensorBuffer = coloredTensor.bufferSync();

                shapes.sort((a,b)=>a.length > b.length).forEach((shapeInfo, index) => {
                  const color = Colors[index % Colors.length]; // Random color for each shape

                  shapeInfo.shape.forEach(([x, y]) => {
                    if (x < width && y < height) {
                      coloredTensorBuffer.set(color[0], y+centeredY-50, x+centeredX-50, 0);
                      coloredTensorBuffer.set(color[1], y+centeredY-50, x+centeredX-50, 1);
                      coloredTensorBuffer.set(color[2], y+centeredY-50, x+centeredX-50, 2);
                    }
                  });
                });
            

                return coloredTensorBuffer.toTensor().mul(255);
              }

              // Function to check if shapes meet the size threshold
              function checkShapesSize(shapes, threshold) {
                for (let shape of shapes) {
                  if (shape >= threshold) {
                    return true;
                  }
                }
                return false;
              }

              // Main process
              const squareSize = 100; // Size of the square
              const sizeThreshold = 5000; // Minimum size of shape to be considered
              const centerOfMass = [
                Math.round(centroidX),
                Math.round(centroidY),
              ];

              const square = extractSquare(maskArray, centerOfMass, squareSize);
              const shapes = detectShapes(square, sizeThreshold);

              let isValidCenter =
                shapes.exceededThreshold === true
                  ? true
                  : checkShapesSize(shapes.shapes, sizeThreshold);

              let coloredTensor = null;
              if (!isValidCenter) {
                coloredTensor = createColoredTensor(shapes.shapes, 480, 640, centerOfMass[0], centerOfMass[1]);
              }

              if (centroidY < centerY - videoHeight * 0.01) {
                debounceTts("road might be ended please be careful");
              } else {
                if (centroidX < centerX - videoWidth * 0.3) {
                  debounceTts("road is turning left please be careful");
                }
                if (centroidY > centerY + videoWidth * 0.3) {
                  debounceTts("road is turning right please be careful");
                }
              }

              const blueMask = tf.tidy(() => {
                const zeros = tf.zerosLike(mask2d);
                const ones = tf.onesLike(mask2d);
                const blueChannel = ones;
                const alphaChannel = mask2d.mul(127).add(128);

                const squareSize = 20;
                const xStart = Math.floor(centroidX);
                const yStart = Math.floor(centroidY);
                const yStart2 = Math.floor(centerY);
                const xStart2 = Math.floor(centerX);

                const blendTensors = (tensorA, tensorB, alpha) => {
                  return tf.tidy(() => {
                    const blendedTensor = tensorA
                      .mul(alpha)
                      .add(tensorB.mul(1 - alpha));
                    return blendedTensor;
                  });
                };
                const addAlphaChannel = (rgbTensor, alphaChannel) => {
                  return tf.tidy(() => {
                    // Ensure the alpha channel has shape [480, 640, 1]
                    const alphaChannelReshaped = alphaChannel.reshape([
                      480, 640, 1,
                    ]);

                    // Concatenate along the last axis to form the RGBA tensor
                    const rgbaTensor = tf.concat(
                      [rgbTensor, alphaChannelReshaped],
                      -1
                    );

                    return rgbaTensor;
                  });
                };
                const detectionBoundaryBox = 100;

                // Create the green square
                let greenChannel = zeros.clone().bufferSync();
                if (isValidCenter) {
                  for (let i = yStart; i < yStart + squareSize; i++) {
                    for (let j = xStart; j < xStart + squareSize; j++) {
                      greenChannel.set(255, i, j);
                    }
                  }
                }
                const squareLeftBorder = Math.max(
                  xStart - detectionBoundaryBox,
                  0
                );
                const squareRightBorder = Math.min(
                  xStart + detectionBoundaryBox,
                  greenChannel.shape[1]
                );
                const squareTopBorder = Math.max(
                  yStart - detectionBoundaryBox,
                  0
                );
                const squareBottomBorder = Math.min(
                  yStart + detectionBoundaryBox,
                  greenChannel.shape[0]
                );
                for (let i = squareTopBorder; i < squareTopBorder + 5; i++) {
                  for (let j = squareLeftBorder; j < squareRightBorder; j++) {
                    greenChannel.set(255, i, j);
                  }
                }
                for (
                  let i = squareBottomBorder - 5;
                  i < squareBottomBorder;
                  i++
                ) {
                  for (let j = squareLeftBorder; j < squareRightBorder; j++) {
                    greenChannel.set(255, i, j);
                  }
                }
                for (let i = squareTopBorder; i < squareBottomBorder; i++) {
                  for (
                    let j = squareLeftBorder;
                    j < squareLeftBorder + 5;
                    j++
                  ) {
                    greenChannel.set(255, i, j);
                  }
                }
                for (let i = squareTopBorder; i < squareBottomBorder; i++) {
                  for (
                    let j = squareRightBorder - 5;
                    j < squareRightBorder;
                    j++
                  ) {
                    greenChannel.set(255, i, j);
                  }
                }

                greenChannel = greenChannel.toTensor();

                // Create the red square
                let redChannel = zeros.clone().bufferSync();
                for (let i = yStart2; i < yStart2 + squareSize; i++) {
                  for (let j = xStart2; j < xStart2 + squareSize; j++) {
                    redChannel.set(255, i, j);
                  }
                }

                redChannel = redChannel.toTensor();
                if (isValidCenter) {
                  redChannel = tf.add(redChannel, greenChannel).div(255);
                }

                // Combine channels
                let mask = tf.stack(
                  [redChannel, greenChannel, blueChannel],
                  -1
                );
                if(coloredTensor) {
                  mask = blendTensors(mask, coloredTensor, 0.5);
                }





                const t2 = tf.browser.fromPixels(canvas);

                const ttt = blendTensors(mask, t2, 0.5);
     

                const ttt2 = addAlphaChannel(ttt, alphaChannel);

                return ttt2;
              });

              // Convert to uint8 since toPixels expects integers
              const blueMaskUint8 = blueMask.cast("int32");

              // Overlay the mask on the canvas
              tf.browser.toPixels(blueMaskUint8, canvas2).then(() => {});
            });
          })
          .catch((error) =>
            console.error("Prediction or post-processing error:", error)
          );
      });

      requestAnimationFrame(drawFrame);
    };

    drawFrame();
  };

  return (
    <div>
      <video ref={videoRef} style={{ display: "none" }}></video>
      <canvas ref={canvasRef}></canvas>
      <canvas ref={canvas2Ref}></canvas>
      <DetectModel />
    </div>
  );
};

export default App;
