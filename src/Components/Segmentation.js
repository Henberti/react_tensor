import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import tts from "./Tts";
import "./Segment.css";

const Segmentation = () => {
    const webcamRef = useRef(null);
    const canvasRef = useRef(null);
    const canvas2Ref = useRef(null);
    const [model, setModel] = useState(null);
    const lastTtsCall = useRef(Date.now());
    const [started, setStarted] = useState(false);

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
        if (!model || !webcamRef.current || !started) return;
        const captureAndPredict = async () => {
            if (
                typeof webcamRef.current !== "undefined" &&
                webcamRef.current !== null &&
                webcamRef.current.video.readyState === 4
            ) {
                const video = webcamRef.current.video;
                const videoWidth = webcamRef.current.video.videoWidth;
                const videoHeight = webcamRef.current.video.videoHeight;

                canvasRef.current.width = videoWidth;
                canvasRef.current.height = videoHeight;

                const context = canvasRef.current.getContext("2d");

                const drawFrame = () => {
                    tf.tidy(() => {
                        context.drawImage(video, 0, 0, videoWidth, videoHeight);

                        const tensCanvas = tf.browser.fromPixels(canvasRef.current);
                        const tensor = tensCanvas
                            .resizeNearestNeighbor([256, 256])
                            .toFloat()
                            .expandDims(0)
                            .div(255);

                        model
                            .predict(tensor)
                            .data()
                            .then((predictionData) => {
                                tf.tidy(() => {
                                    let maskTensor = tf.tensor2d(predictionData, [256, 256]);
                                    maskTensor = maskTensor.expandDims(-1);
                                    const resizedMask = maskTensor.resizeNearestNeighbor([videoHeight, videoWidth]);
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
                                    const centroidY = (sumY / count) * (videoHeight / mask2d.shape[0]);

                                    const centerX = videoWidth / 2;
                                    const centerY = videoHeight / 2;

                                    if (centroidY < centerY - videoHeight * 0.3) {
                                        debounceTts("road might be ended please be careful");
                                    } else {
                                        if (centroidX < centerX - videoWidth * 0.3) {
                                            debounceTts("road is turning left please be careful");
                                        }
                                        if (centroidX > centerX + videoWidth * 0.3) {
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
                                                const alphaChannelReshaped = alphaChannel.reshape([
                                                    480, 640, 1,
                                                ]);

                                                const rgbaTensor = tf.concat(
                                                    [rgbTensor, alphaChannelReshaped],
                                                    -1
                                                );

                                                return rgbaTensor;
                                            });
                                        };

                                        let greenChannel = zeros.clone().bufferSync();
                                        for (let i = yStart; i < yStart + squareSize; i++) {
                                            for (let j = xStart; j < xStart + squareSize; j++) {
                                                greenChannel.set(255, i, j);
                                            }
                                        }
                                        greenChannel = greenChannel.toTensor();

                                        let redChannel = zeros.clone().bufferSync();
                                        for (let i = yStart2; i < yStart2 + squareSize; i++) {
                                            for (let j = xStart2; j < xStart2 + squareSize; j++) {
                                                redChannel.set(255, i, j);
                                            }
                                        }
                                        redChannel = redChannel.toTensor();

                                        const mask = tf.stack(
                                            [redChannel, greenChannel, blueChannel],
                                            -1
                                        );
                                        const t2 = tf.browser.fromPixels(canvasRef.current);

                                        const ttt = blendTensors(mask, t2, 0.5);
                                        const ttt2 = addAlphaChannel(ttt, alphaChannel);

                                        return ttt2;
                                    });

                                    const blueMaskUint8 = blueMask.cast("int32");

                                    tf.browser.toPixels(blueMaskUint8, canvas2Ref.current).then(() => { });
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
        };

        captureAndPredict();
    }, [model, started]);

    const videoConstraints = {
        width: 640,
        height: 480,
        facingMode: "environment"
    };

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
                <canvas
                    ref={canvas2Ref}
                    style={{
                        position: "absolute",
                        marginLeft: "auto",
                        marginRight: "auto",
                        left: 0,
                        right: 0,
                        textAlign: "center",
                        zindex: 7,
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
        </div>
    );
};

export default Segmentation;
