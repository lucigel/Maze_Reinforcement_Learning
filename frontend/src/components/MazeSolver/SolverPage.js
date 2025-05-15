import React, { useEffect, useState } from 'react';
import { useMazeContext } from '../../context/MazeContext';
import useMazeSolver from '../../hooks/useMazeSolver';
import ModelSelector from './ModelSelector';
import MazeSelector from './MazeSelector';
import MazeVisualization from './MazeVisualization';
import MouseVisualization from './MouseVisualization';
import ValueFunctionHeatmap from './ValueFunctionHeatmap';
import PerformanceStats from './PerformanceStats';
import StepControls from '../MazeGenerator/StepControls';
import { toast } from 'react-toastify';
import '../../styles/MazeSolver.css';

const SolverPage = () => {
  const {
    maze,
    solverAlgorithm, setSolverAlgorithm,
    simulationSpeed, setSimulationSpeed,
    isSolving, showHeatmap, setShowHeatmap,
    qValues, path
  } = useMazeContext();

  const [apiStatus, setApiStatus] = useState('checking');

  const {
    solveMaze,
    resetSimulation,
    startAnimation,
    stopAnimation,
    skipToEnd,
    nextStep,
    prevStep,
    getCurrentMousePosition,
    isAnimating,
    currentPathIndex,
    totalSteps,
    errorMessage,
    solutionStats
  } = useMazeSolver();

  // Kiểm tra kết nối API khi trang được tải
  useEffect(() => {
    const checkApi = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        const data = await response.json();
        
        if (data.status === 'ok') {
          setApiStatus('connected');
        } else {
          setApiStatus('error');
          toast.error('Không thể kết nối đến API backend!');
        }
      } catch (error) {
        console.error('Lỗi khi kiểm tra API:', error);
        setApiStatus('error');
        toast.error('Không thể kết nối đến API backend!');
      }
    };
    
    checkApi();
  }, []);

  // Xử lý thay đổi thuật toán
  const handleAlgorithmChange = (e) => {
    setSolverAlgorithm(e.target.value);
    resetSimulation();
  };

  // Xử lý thay đổi tốc độ mô phỏng
  const handleSpeedChange = (e) => {
    setSimulationSpeed(parseInt(e.target.value, 10));
  };

  // Xử lý chuyển đổi hiển thị heatmap
  const toggleHeatmap = () => {
    setShowHeatmap(!showHeatmap);
  };

  // Xử lý việc bắt đầu giải mê cung
  const handleSolveMaze = async () => {
    if (!maze) {
      toast.warning('Vui lòng chọn hoặc tạo mê cung trước khi giải!');
      return;
    }
    
    try {
      await solveMaze();
    } catch (error) {
      console.error('Lỗi khi giải mê cung:', error);
      toast.error(`Lỗi khi giải mê cung: ${error.message || 'Không xác định'}`);
    }
  };

  // Hiển thị vị trí hiện tại của chuột
  const mousePosition = getCurrentMousePosition();

  return (
    <div className="solver-page">
      <h2>Giải mê cung với Học tăng cường</h2>
      
      {apiStatus === 'error' && (
        <div className="api-error-message">
          <p>⚠️ Không thể kết nối đến API backend. Vui lòng đảm bảo rằng server đang chạy tại http://localhost:8000</p>
        </div>
      )}
      
      <div className="solver-layout">
        <div className="control-panel">
          <MazeSelector />
          
          <ModelSelector
            algorithm={solverAlgorithm}
            onAlgorithmChange={handleAlgorithmChange}
            onSpeedChange={handleSpeedChange}
            speed={simulationSpeed}
            disabled={isSolving}
          />
          
          <button
            className="btn btn-primary solve-btn"
            onClick={handleSolveMaze}
            disabled={isSolving || !maze || apiStatus === 'error'}
          >
            {isSolving ? 'Đang giải...' : 'Giải mê cung'}
          </button>
          
          {errorMessage && (
            <div className="error-message">
              <p>{errorMessage}</p>
            </div>
          )}
          
          {isSolving && (
            <StepControls
              onNext={nextStep}
              onPrev={prevStep}
              onSkipToEnd={skipToEnd}
              onStop={stopAnimation}
              onStart={startAnimation}
              isAnimating={isAnimating}
              currentStep={currentPathIndex + 1}
              totalSteps={totalSteps}
              disabled={!isSolving}
            />
          )}
          
          {qValues && (
            <div className="heatmap-toggle">
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={toggleHeatmap}
                />
                <span className="toggle-slider"></span>
              </label>
              <span>Hiển thị Heatmap giá trị Q</span>
            </div>
          )}
          
          {solutionStats && <PerformanceStats stats={solutionStats} />}
        </div>
        
        <div className="visualization-panel">
          <div className="maze-visualization-container">
            {maze ? (
              <>
                <MazeVisualization />
                {path && path.length > 0 && (
                  <MouseVisualization position={mousePosition} />
                )}
              </>
            ) : (
              <div className="no-maze-message">
                <p>Vui lòng chọn hoặc tạo mê cung trước</p>
              </div>
            )}
          </div>
          
          {showHeatmap && qValues && (
            <ValueFunctionHeatmap qValues={qValues} maze={maze} />
          )}
        </div>
      </div>
    </div>
  );
};

export default SolverPage;