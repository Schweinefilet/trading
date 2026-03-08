import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import StockDetail from './pages/StockDetail';
import Portfolio from './pages/Portfolio';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout><Dashboard /></Layout>} />
        <Route path="/stock/:ticker" element={<StockDetail />} />
        <Route path="/portfolio" element={<Layout><Portfolio /></Layout>} />
      </Routes>
    </Router>
  );
}

export default App;
