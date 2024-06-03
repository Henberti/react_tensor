import React, { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import { useSegmentation, useDetection } from "../hooks/modelsHook";
//this is the model that we will use to predict the mask hen
const App = () => {
  const videoRef = useRef(null);
  const wasRendered = useRef(false);
  const canvas2Ref = useRef();
  const { model, getSegmentation } = useSegmentation(
    "models/jsconv2/model.json"
  );

  const { detect, model: detectionModel } = useDetection();

  useEffect(() => {
    if (!model || !videoRef.current) return;
    if (!detectionModel) return;
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
  }, [model, detectionModel]);

  const captureAndPredict = async () => {
    if (!videoRef.current || !model) return;

    const video = videoRef.current;
    const canvas2 = canvas2Ref.current;
    const ctx = canvas2.getContext("2d");

    await detect(video, video.vi);
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    const drawFrame = async () => {
      const detection = await detect(video, videoWidth, videoHeight);
      detection.forEach((prediction) => {
        const [x, y] = prediction.bbox;
        const [width, height] = prediction.bbox.slice(2);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
        ctx.fillStyle = "red";
        ctx.fillText(prediction.class, x, y);
      });
      tf.tidy(() => {
        tf.tidy(() => {
          Promise.all([getSegmentation(ctx, video, videoHeight, videoWidth)])
            .then(([segmentation]) => {
              // console.log('detections', detection)
              tf.tidy(() => {
                tf.browser.toPixels(segmentation.blueMaskUint8, canvas2);
                tf.dispose(segmentation.blueMaskUint8);
              });
            })
            .catch((error) => {
              console.error("Error getting segmentation:", error);
            });
        });
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
