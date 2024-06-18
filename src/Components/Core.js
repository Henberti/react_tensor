import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { useSegmentation, useDetection } from "../hooks/modelsHook";
import { useTts } from "../hooks/ttsHook";
import Button from "./Button";

const Core = ({ mode }) => {
  const { addMessage, tts } = useTts();
  const videoRef = useRef(null);
  const wasRendered = useRef(false);
  const canvas2Ref = useRef();
  const [isStarted, setIsStarted] = useState(false);
  const { model, getSegmentation } = useSegmentation(
    "models/jsconv5/model.json"
  );
  const { detect, model: detectionModel } = useDetection();

  function isMobile() {
    return /Mobi|Android/i.test(navigator.userAgent);
  }

  useEffect(() => {
    if (!model || !videoRef.current || !detectionModel || wasRendered.current)
      return;

    if (isStarted) {
      wasRendered.current = true;

      if (
        canvas2Ref.current &&
        canvas2Ref.current.style.display === "none" &&
        mode === "Visual"
      ) {
        canvas2Ref.current.style.display = "block";
      }

      navigator.mediaDevices
        .getUserMedia({
          video: isMobile() ? { facingMode: { exact: "environment" } } : true,
        })
        .then((stream) => {
          videoRef.current.srcObject = stream;
          videoRef.current.play().then(() => {
            captureAndPredict();
          });
        })
        .catch((error) => {
          console.error("Error accessing webcam:", error);
        });
    }
  }, [model, detectionModel, isStarted]);

  useEffect(() => {
    if (!isStarted && wasRendered.current) {
      videoRef.current.srcObject.getTracks().forEach((track) => {
        track.stop();
      });
      wasRendered.current = false;
      if (canvas2Ref.current) {
        canvas2Ref.current.style.display = "none";
      }
    }
  }, [isStarted]);

  const captureAndPredict = async () => {
    if (!videoRef.current || !model) return;

    const video = videoRef.current;
    const canvas2 = canvas2Ref.current;
    const ctx = canvas2.getContext("2d");

 
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    const drawFrame = async () => {
      if (!wasRendered.current) return;
      const detection = await detect(video, videoWidth, videoHeight);
      detection.forEach((prediction) => {
        addMessage(prediction.class, "obstacle");
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
      if (isStarted) {
        requestAnimationFrame(drawFrame);
      }
    };

    drawFrame();
  };

  const onToggle = (operation) => {
    if (operation === 1) {
      tts(
        "Welcome to SafePath, please hold your phone in front of you and start walking."
      );
    }

    setIsStarted(operation === 1);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Button onClick={onToggle} />

      <video
        ref={videoRef}
        hidden={mode === "Demo"}
        style={{ display: "none" }}
      ></video>
      <canvas ref={canvas2Ref} hidden={mode === "Demo"}></canvas>
    </div>
  );
};

export default Core;
