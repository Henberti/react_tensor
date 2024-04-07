import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';


const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef();
  const wasRendered = useRef(false);
  const canvas2Ref = useRef();
  const [model, setModel] = useState(null);

  useEffect(() => {
    tf.loadLayersModel(process.env.PUBLIC_URL + '/jsconv/model.json')
      .then((loadedModel) => {
        console.log('Model loaded successfully:', loadedModel);
        setModel(loadedModel);
      })
      .catch((error) => {
        console.error('Error loading model:', error);
      });
  }, []);

  useEffect(() => {
    if (!model || !videoRef.current) return;
    if (wasRendered.current) return;
    wasRendered.current = true;
    
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
        videoRef.current.play().then(() => {
          captureAndPredict();
        });
      })
      .catch((error) => {
        console.error('Error accessing webcam:', error);
      });
  }, [model]);

  const captureAndPredict = async () => {
    if (!videoRef.current || !model) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const canvas2 = canvas2Ref.current;

    const drawFrame = () => {
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
      canvas.width = videoWidth;
      canvas.height = videoHeight;

      context.drawImage(video, 0, 0, videoWidth, videoHeight);
      
      // Preprocess the frame
      const tensor = tf.browser.fromPixels(canvas).resizeNearestNeighbor([256, 256]).toFloat().expandDims(0).div(255);
      tensor.norm('euclidean', 0, 1)
     
      
      
      model.predict(tensor).data().then(predictionData => {
        tf.tidy(() => {
          // Assuming predictionData is a flat array; convert to a 2D tensor (256x256) as before
          let maskTensor = tf.tensor2d(predictionData, [256, 256]);
      
          // Expand dimensions to make it [256, 256, 1], which is necessary for resizeNearestNeighbor
          maskTensor = maskTensor.expandDims(-1);
      
          // Now resize the mask to match the video frame dimensions
          const resizedMask = maskTensor.resizeNearestNeighbor([videoHeight, videoWidth]);
      
          // Remove the singleton dimension after resizing so we can use it for masking
          let mask2d = resizedMask.squeeze();
          mask2d = mask2d.greater(0.5).cast('float32');
          // mask2d.print();

      
          // Create a blue mask with opacity
          const blueMask = tf.tidy(() => {
            const zeros = tf.zerosLike(mask2d);
            const ones = tf.onesLike(mask2d);
            const blueChannel = ones.mul(255); // Full intensity for blue channel
            const alphaChannel = mask2d.mul(127).add(128); // Adjust opacity
            return tf.stack([zeros, zeros, blueChannel, alphaChannel], -1);
          });
          // blueMask.print();
      
          // Convert to uint8 since toPixels expects integers
          const blueMaskUint8 = blueMask.cast('int32');
      
          // Overlay the mask on the canvas
          tf.browser.toPixels(blueMaskUint8, canvas2).then(() => blueMaskUint8.dispose());
        });
      }).then(()=>model.cleanUp())
      .catch(error => console.error('Prediction or post-processing error:', error));
      
      requestAnimationFrame(drawFrame);
    };

    drawFrame();
  };

  return (
    <div>
      <video ref={videoRef} style={{display: 'none'}}></video>
      <canvas ref={canvasRef}></canvas>
      <canvas  ref={canvas2Ref}></canvas>
    </div>
  );
};

export default App;
