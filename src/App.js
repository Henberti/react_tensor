import React from "react";
// import DetectModel from "./Components/DetectModel";
import Segmentation from "./Components/Segmentation";
import Core from "./Components/Core";
import Navbar from "./Components/Navbar"
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";


const App = () => {

  return (
    <div>
      <Navbar />
      <Routes>
        <Route path="/demo" element={<Core mode="Demo" />} />       
         <Route path="/visual" element={<Core mode="Visual" />} />
      </Routes>
    </div>
  );
}

export default App;
