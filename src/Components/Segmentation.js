import React, { useRef, useEffect } from "react";
import {
  drawMask,
  createColoredTensor,
} from "../utils/drawFunctions";
import * as tf from "@tensorflow/tfjs";
import tts from "../Components/Tts";
import { useSegmentation } from "../hooks/modelsHook";
//this is the model that we will use to predict the mask hen
const App = () => {
  const videoRef = useRef(null);
  const wasRendered = useRef(false);
  const canvas2Ref = useRef();
  const { start, model } = useSegmentation("models/jsconv2/model.json");
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
    const canvas2 = canvas2Ref.current;
    const ctx = canvas2.getContext("2d");

    const drawFrame = () => {
      tf.tidy(() => {
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;

        start(video)
          .then((res) => {
            tf.tidy(() => {
              const centroidY = res.centerOfMass[1];
              const centroidX = res.centerOfMass[0];
              const centerY = res.videoCenter[1];
              const centerX = res.videoCenter[0];

              let coloredTensor = null;

              if (res.shapes) {
                coloredTensor = createColoredTensor(
                  res.shapes,
                  videoHeight,
                  videoWidth,
                  centroidX,
                  centroidY
                );
              }
              //move this to a hook or function
              // if (centroidY < centerY - videoHeight * 0.01) {
              //   debounceTts("road might be ended please be careful");
              // } else {
              //   if (centroidX < centerX - videoWidth * 0.3) {
              //     debounceTts("road is turning left please be careful");
              //   }
              //   if (centroidY > centerY + videoWidth * 0.3) {
              //     debounceTts("road is turning right please be careful");
              //   }
              // }

              const blueMask = tf.tidy(() => {
                return drawMask(
                  ctx,
                  res.mask2d,
                  res.isValidCenter,
                  video,
                  coloredTensor,
                  centroidX,
                  centroidY,
                  videoWidth,
                  videoHeight
                );
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
      <canvas ref={canvas2Ref}></canvas>
    </div>
  );
};

export default App;
