import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
// Components
import Header from './components/Common/Header';
import Footer from './components/Common/Footer';
import Navigation from './components/Common/Navigation';

// Pages
import GeneratorPage from './components/MazeGenerator/GeneratorPage';
import SolverPage from './components/MazeSolver/SolverPage';

// Context
import { MazeProvider } from './context/MazeContext';

// Styles
import './styles/App.css';

function App() {
  return (
    <MazeProvider>
      <Router>
        <div className="app-container">
          <Header />
          <Navigation />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<GeneratorPage />} />
              <Route path="/solver" element={<SolverPage />} />
            </Routes>
          </main>
          <Footer />
          <ToastContainer position="bottom-right" />
        </div>
      </Router>
    </MazeProvider>
  );
}

export default App;