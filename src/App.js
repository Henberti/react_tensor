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

                // Create the green square
                let greenChannel = zeros.clone().bufferSync();
                for (let i = yStart; i < yStart + squareSize; i++) {
                  for (let j = xStart; j < xStart + squareSize; j++) {
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

                // Combine channels
                const mask = tf.stack(
                  [redChannel, greenChannel, blueChannel],
                  -1
                );
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
