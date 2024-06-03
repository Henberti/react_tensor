export const drawRect = (prediction, ctx) => {
    const [x, y, width, height] = prediction.bbox;
    const text = prediction.class;
  
    // Styling
    const color = "red";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.font = "18px Arial";
    ctx.fillStyle = color;
  
    // Draw rectangle and text
    ctx.beginPath();
    ctx.fillText(text, x, y > 10 ? y - 5 : 10);
    ctx.rect(x, y, width, height);
    ctx.stroke();
  };
  