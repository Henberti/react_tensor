import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocossd from "@tensorflow-models/coco-ssd"
import Webcam from "react-webcam";
import "./App.css";
import tts from "./Components/Tts";
import { drawRect } from "./utils";


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
    facingMode: "environment"
  };

  const runCoco = async () => {
    const net = await cocossd.load();
    setInterval(() => {
      detect(net);
    }, 10);
  };

  const detectionHistory = {};

  const updateDetectionHistory = (predictions) => {
    const detectedClasses = predictions.map(p => p.class);
    detectedClasses.forEach(cls => {
      if (!detectionHistory[cls]) {
        detectionHistory[cls] = 1;
      } else {
        detectionHistory[cls]++;
      }
    });
    Object.keys(detectionHistory).forEach(cls => {
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

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;


      const obj = await net.detect(video);
      updateDetectionHistory(obj);
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, videoWidth, videoHeight);

      const frameCenter = { x: videoWidth / 2, y: videoHeight / 2 };

      obj.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const bboxCenter = { x: x + width / 2, y: y + height / 2 };

        if (isCloseToCenter(frameCenter, bboxCenter) && checkThreshold(prediction.class)) {
          const messages = [
            "There is a " + prediction.class + " in front of you.",
            "you are walking toward a " + prediction.class,
            "There is a " + prediction.class + " in your way please adjust your path.",
          ];
          const randomIndex = Math.floor(Math.random() * messages.length);
          const message = messages[randomIndex];
          debounceTts(message);
          drawRect(prediction, ctx);
        }

      });
    };
  };

  function isCloseToCenter(frameCenter, bboxCenter) {
    const distance = Math.sqrt(Math.pow(frameCenter.x - bboxCenter.x, 2) + Math.pow(frameCenter.y - bboxCenter.y, 2));
    const threshold = 70;
    return distance < threshold;
  }

  useEffect(() => {
    if (started) runCoco()
  }, [started]);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          muted={true}
          audio={false}
          videoConstraints={videoConstraints}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 8,
            width: 640,
            height: 480,
          }}
        />
      </header>
      <button onClick={() => {
        let voices = window.speechSynthesis.getVoices();
        const voice = voices[1];
        tts("Started", voice);
        setStarted(true);
      }}>Start</button>
      {/* <Dictaphone setStarted={setStarted} /> */}
    </div>
  );
}

export default DetectModel;
