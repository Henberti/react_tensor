import { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocossd from "@tensorflow-models/coco-ssd";
import {
  ResizeMask,
  filterMaskPrediction,
  returnCentroid,
  extractSquare,
  detectShapes,
  checkShapesSize,
} from "../utils/segmentationFunctions";
import { createColoredTensor, drawMask } from "../utils/drawFunctions";
import { useTts } from './ttsHook';

const isCloseToCenter = (frameCenter, bboxCenter, threshold) => {
  const distance = Math.sqrt(
    Math.pow(frameCenter.x - bboxCenter.x, 2) +
    Math.pow(frameCenter.y - bboxCenter.y, 2)
  );
  return distance < threshold;
}

const useSegmentation = (
  modelPath,
  maskThreshold = 0.5,
  squareSize = 100,
  squareThreshold = 5000
) => {
  const [model, setModel] = useState(null);
  const { addMessage } = useTts();

  useEffect(() => {
    tf.loadLayersModel(process.env.PUBLIC_URL + modelPath)
      .then((loadedModel) => {
        console.log("Model loaded successfully:", loadedModel);
        setModel(loadedModel);
      })
      .catch((error) => {
        console.error("Error loading model:", error);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = async (video) => {
    const errorsArray = [
      model ? "" : "Model is not loaded",
      video ? "" : "No video element",
    ];
    if (errorsArray.some((v) => v !== "")) {
      const errorsString = errorsArray.join("\n");
      console.log(errorsString);
      return null;
    }
    // Capture the video frame as a tensor
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    const tensCanvas = tf.browser.fromPixels(video);

    const tensor = tf.tidy(() => {
      // Preprocess the tensor
      return tensCanvas
        .resizeNearestNeighbor([256, 256])
        .toFloat()
        .sub(tf.scalar(127.5))
        .div(tf.scalar(127.5))
        .expandDims(0);
    });

    try {
      // Perform the prediction
      const predictionData = await model.predict(tensor).data();

      // Process the prediction data
      const result = tf.tidy(() => {
        // Resize and filter the mask

        let mask = ResizeMask(predictionData, videoHeight, videoWidth);

        mask = filterMaskPrediction(mask, maskThreshold);

        // Extract mask data and calculate centroid
        const maskArray = mask.arraySync();

        const { centroidX, centroidY, centerX, centerY } = returnCentroid(
          maskArray,
          mask.shape,
          videoWidth,
          videoHeight
        );

        // Determine center of mass and extract the relevant square
        const centerOfMass = [Math.round(centroidX), Math.round(centroidY)];

        const extractedSquare = extractSquare(
          maskArray,
          centerOfMass,
          squareSize
        );

        // Detect shapes and determine if the center is valid
        const shapes = detectShapes(extractedSquare, squareThreshold);

        const isValidCenter = shapes.exceededThreshold
          ? true
          : checkShapesSize(shapes.shapes, squareThreshold);

        // Return the final result
        return {
          isValidCenter,
          centerOfMass,
          videoCenter: [Math.floor(centerX), Math.floor(centerY)],
          shapes: shapes.shapes,
          mask2d: mask,
        };
      });

      return result;
    } finally {
      // Ensure the tensor is disposed of
      tensor.dispose();
    }
  };

  const counterRef = useRef(-1)

  const getSegmentation = async (ctx, video, height, width) => {
    return start(video).then((res) => {
      return tf.tidy(() => {
        let coloredTensor = null;
        const centroidX = res.centerOfMass[0];
        const centroidY = res.centerOfMass[1];
        const centerX = width / 2;
        const centerY = height / 2;

        if(res.isValidCenter && counterRef.current < 10){
          counterRef.current++;
        }
        else if(!res.isValidCenter && counterRef.current >0){
          counterRef.current--
        }

        if(counterRef.current ===10){
          addMessage('Safe path detected')
        }
        else if(counterRef.current === 0){
          addMessage('road might be ended please be careful')
        }

        

       

        if (res.shapes) {
          coloredTensor = createColoredTensor(
            res.shapes,
            height,
            width,
            centroidX,
            centroidY
          );
        }
        const blueMask = tf.tidy(() => {
          return drawMask(
            ctx,
            res.mask2d,
            res.isValidCenter,
            video,
            coloredTensor,
            centroidX,
            centroidY,
            width,
            height
          );
        });
        // Convert to uint8 since toPixels expects integers
        const blueMaskUint8 = blueMask.cast("int32");
        return { blueMaskUint8, res };
      });
    });
  };

  return { model, start, getSegmentation };
};

const estimateDistance = (pixelHeight) => {
  if (pixelHeight > 400) return 1; // Very close (around 1 meter)
  if (pixelHeight > 150) return 3; // Close (around 3 meters)
  return 5; // Far (5 meters or more)
};

const useDetection = () => {
  const [model, setModel] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      const loadedModel = await cocossd.load();
      setModel(loadedModel);
    };
    loadModel();
  }, []);

  const detect = async (video, width, height) => {
    const errorsArray = [
      model ? "" : "Model is not loaded",
      video ? "" : "No video element",
    ];
    if (errorsArray.some((v) => v !== "")) {
      const errorsString = errorsArray.join("\n");
      console.log(errorsString);
      return null;
    }
    const frameCenter = { x: width / 2, y: height / 2 };

    return model.detect(video).then((res) => {
      const bboxes = [];
      res.forEach((prediction) => {
        const [x, y, w, h] = prediction.bbox;
        const bboxCenter = { x: x + w / 2, y: y + h / 2 };
        const distance = estimateDistance(h);
        prediction.distance = distance;
        if (isCloseToCenter(frameCenter, bboxCenter, 70)) {
          bboxes.push(prediction);
        }
      });
      return bboxes;
    });
  };

  return { model, detect };
};

export { useSegmentation, useDetection };
