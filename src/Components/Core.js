import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { useSegmentation, useDetection } from "../hooks/modelsHook";
import { useTts } from "../hooks/ttsHook";
import Button from "./Button";
import { PathFinderLoader } from "../Components/Loading";

const Core = ({ mode }) => {
  const { addMessage, Tts } = useTts();
  const videoRef = useRef(null);
  const wasRendered = useRef(false);
  const canvas2Ref = useRef();
  const [isStarted, setIsStarted] = useState(false);
  const detectionArray = useRef([]);
  const { model, getSegmentation } = useSegmentation("models/jsconv8/model.json");
  const { detect, model: detectionModel } = useDetection();
  const objects = ["person", "car", "bicycle", "motorcycle", "chair", "truck", "fire hydrant", "stop sign", "couch", "potted plant", "dining table", "tv"];

  function isMobile() {
    return /Mobi|Android/i.test(navigator.userAgent);
  }

  useEffect(() => {
    const initializeCamera = async () => {
      if (!model || !videoRef.current || !detectionModel || wasRendered.current) return;

      if (isStarted) {
        wasRendered.current = true;

        if (canvas2Ref.current && canvas2Ref.current.style.display === "none" && mode === "Visual") {
          canvas2Ref.current.style.display = "block";
        }

        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: isMobile() ? { facingMode: { exact: "environment" } } : true,
          });
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
     

          captureAndPredict();
        } catch (error) {
          console.error("Error accessing webcam:", error);
        }
      }
    };

    initializeCamera();
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
        const [x, y] = prediction.bbox;
        const [width, height] = prediction.bbox.slice(2);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
        ctx.fillStyle = "red";
        ctx.fillText(prediction.class, x, y);
    
        const distance = prediction.distance;
        if (!detectionArray.current.includes(prediction.class) && objects.includes(prediction.class)) {
          if (distance <= 1) {
            addMessage("Stop, there is a " + prediction.class + " in front of you", "alert", true);
            detectionArray.current.push(prediction.class);
            setTimeout(() => {
              detectionArray.current = detectionArray.current.filter((item) => item !== prediction.class);
            }, 4000);
          } else if (distance > 1 && distance <= 3) {
            addMessage(prediction.class, "obstacle");
            detectionArray.current.push(prediction.class);
            setTimeout(() => {
              detectionArray.current = detectionArray.current.filter((item) => item !== prediction.class);
            }, 4000);
          }
        }
      });
    
      tf.tidy(() => {getSegmentationAndDraw()});
    
      if (isStarted) {
        await tf.nextFrame();
        drawFrame();
      }
    };
    
    const getSegmentationAndDraw = async () => {
      try {
        const segmentation = await getSegmentation(ctx, video, videoHeight, videoWidth, mode === "Visual");
        if (mode !== "Visual") return;
        await tf.browser.toPixels(segmentation.blueMaskUint8, canvas2);
        tf.dispose(segmentation.blueMaskUint8);
      } catch (error) {
        console.error("Error getting segmentation:", error);
      }
    };
    
    drawFrame();
    
  };

  const onToggle = (operation) => {
    if (operation === 1) {
      Tts("Welcome to SafePath, please hold your camera in front of you and start walking.");
    }
    setIsStarted(operation === 1);
  };

  return !model ? (
    <PathFinderLoader />
  ) : (
    <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
      {model && <Button onClick={onToggle} />}
      <video ref={videoRef} hidden={mode === "Demo"} style={{ display: "none" }}></video>
      <canvas ref={canvas2Ref} hidden={mode === "Demo"}></canvas>
    </div>
  );
};

export default Core;
