import "./buttonStyle.css";
import React, { useState } from "react";
const buttonColor = ["#28a745", "#dc3545"];
const buttonText = ["Start", "Stop"];


const Button = ({ onClick }) => {
  const [click, setClick] = useState(0);

  const handleClick = () => {
    const newStateClick = click === 0 ? 1 : 0;
    setClick(newStateClick)
    onClick(newStateClick);
  };

  return (
    <button
      onClick={handleClick}
      style={{ backgroundColor: buttonColor[click] }}
      className="btn-safe"
    >
      {buttonText[click]}
    </button>
  );
};
export default Button;
