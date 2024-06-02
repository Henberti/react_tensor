import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import {
  ResizeMask,
  filterMaskPrediction,
  returnCentroid,
  extractSquare,
  detectShapes,
  checkShapesSize,
} from "../utils/segmentationFunctions";

const useSegmentation = (
  modelPath,
  maskThreshold = 0.5,
  squareSize = 100,
  squareThreshold = 5000
) => {
  const [model, setModel] = useState(null);

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
          mask2d:mask
        };
      });

      return result;
    } finally {
      // Ensure the tensor is disposed of
      tensor.dispose();
    }
  };

  return { model, start };
};

export { useSegmentation };
