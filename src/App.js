import React from "react";
// import DetectModel from "./Components/DetectModel";
import Segmentation from "./Components/Segmentation";
import Navbar from "./Components/Navbar"
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";


const App = () => {

  return (
    <div>
      <Navbar />
      <Routes>
        <Route path="/visual" element={<Segmentation />} />
        {/* <Route path="/terrainsegmentation" element={<Segmentation />} /> */}
      </Routes>
    </div>
  );
}

export default App;
