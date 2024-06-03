import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocossd from "@tensorflow-models/coco-ssd";
import Webcam from "react-webcam";
import "./Detect.css";
import tts from "./Tts";
import { drawRect } from "../utils";

function DetectModel() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [started, setStarted] = useState(false);
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

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "environment",
  };

  const runCoco = async () => {
    const net = await cocossd.load();
    setInterval(() => {
      detect(net);
    }, 10);
  };

  const detectionHistory = {};

  const updateDetectionHistory = (predictions) => {
    const detectedClasses = predictions.map((p) => p.class);
    detectedClasses.forEach((cls) => {
      if (!detectionHistory[cls]) {
        detectionHistory[cls] = 1;
      } else {
        detectionHistory[cls]++;
      }
    });
    Object.keys(detectionHistory).forEach((cls) => {
      if (!detectedClasses.includes(cls)) {
        detectionHistory[cls] = 0;
      }
    });
  };

  const checkThreshold = (className, threshold = 4) => {
    return detectionHistory[className] >= threshold;
  };

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video and canvas dimensions
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // Get detections
      const obj = await net.detect(video);

      // Update detection history
      updateDetectionHistory(obj);

      // Draw bounding boxes
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, videoWidth, videoHeight); // Clear previous drawings

      const frameCenter = { x: videoWidth / 2, y: videoHeight / 2 };

      obj.forEach((prediction) => {
        const [x, y, width, height] = prediction.bbox;
        const bboxCenter = { x: x + width / 2, y: y + height / 2 };


        // Draw bounding boxes if the object is close to the center and threshold met
        if (isCloseToCenter(frameCenter, bboxCenter) && checkThreshold(prediction.class)) {
          const messages = [
            "There is a " + prediction.class + " in front of you.",
            "You are walking toward a " + prediction.class,
            "There is a " + prediction.class + " in your way. Please adjust your path.",
          ];
          const randomIndex = Math.floor(Math.random() * messages.length);
          const message = messages[randomIndex];
          debounceTts(message);
          drawRect(prediction, ctx); // Draw bounding box
        }
      });
    }
  };

  function isCloseToCenter(frameCenter, bboxCenter) {
    const distance = Math.sqrt(Math.pow(frameCenter.x - bboxCenter.x, 2) + Math.pow(frameCenter.y - bboxCenter.y, 2));
    const threshold = 70;
    return distance < threshold;
  }

  useEffect(() => {
    if (started) runCoco();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [started]);

  return (
    <div className="App">
      <header className="App-header">
        <div className="webcam-canvas-container">
          <Webcam
            ref={webcamRef}
            muted={true}
            audio={false}
            videoConstraints={videoConstraints}
            className="webcam"
          />
          <canvas ref={canvasRef} className="canvas" />
        </div>
      </header>
      <button
        onClick={() => {
          let voices = window.speechSynthesis.getVoices();
          const voice = voices[1];
          tts("Started", voice);
          setStarted(true);
        }}
      >
        Start
      </button>
    </div>
  );
}

export default DetectModel;
