import React, { createContext, useState, useContext } from 'react';

// Tạo Context
const MazeContext = createContext();

// Constants
export const PATH = 0;
export const WALL = 1;
export const START = 2;
export const GOAL = 3;

// Hook để sử dụng context
export const useMazeContext = () => useContext(MazeContext);

// Provider Component
export const MazeProvider = ({ children }) => {
  // State cho mê cung
  const [maze, setMaze] = useState(null);
  const [mazeSize, setMazeSize] = useState({ width: 10, height: 10 });
  const [startPos, setStartPos] = useState(null);
  const [goalPos, setGoalPos] = useState(null);
  const [generatorAlgorithm, setGeneratorAlgorithm] = useState('dfs');
  const [solverAlgorithm, setSolverAlgorithm] = useState('q_learning');
  const [generationSpeed, setGenerationSpeed] = useState(50); // ms delay giữa các bước
  const [simulationSpeed, setSimulationSpeed] = useState(100); // ms delay giữa các bước
  const [path, setPath] = useState([]);
  const [qValues, setQValues] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSolving, setIsSolving] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);

  // Phương thức lưu mê cung vào localStorage
  const saveMaze = () => {
    if (!maze) return false;
    
    const mazeData = {
      maze,
      mazeSize,
      startPos,
      goalPos,
      generatorAlgorithm,
      timestamp: Date.now()
    };
    
    try {
      // Lấy danh sách mê cung đã lưu
      const savedMazes = JSON.parse(localStorage.getItem('savedMazes')) || [];
      
      // Thêm mê cung mới
      savedMazes.push(mazeData);
      
      // Lưu lại, giới hạn 10 mê cung để tránh quá tải localStorage
      localStorage.setItem('savedMazes', JSON.stringify(savedMazes.slice(-10)));
      
      return true;
    } catch (error) {
      console.error('Lỗi khi lưu mê cung:', error);
      return false;
    }
  };

  // Phương thức lấy danh sách mê cung đã lưu
  const getSavedMazes = () => {
    try {
      return JSON.parse(localStorage.getItem('savedMazes')) || [];
    } catch (error) {
      console.error('Lỗi khi lấy danh sách mê cung:', error);
      return [];
    }
  };

  // Phương thức load mê cung từ localStorage
  const loadMaze = (index) => {
    try {
      const savedMazes = getSavedMazes();
      if (index < 0 || index >= savedMazes.length) return false;
      
      const mazeData = savedMazes[index];
      setMaze(mazeData.maze);
      setMazeSize(mazeData.mazeSize);
      setStartPos(mazeData.startPos);
      setGoalPos(mazeData.goalPos);
      setGeneratorAlgorithm(mazeData.generatorAlgorithm);
      
      return true;
    } catch (error) {
      console.error('Lỗi khi load mê cung:', error);
      return false;
    }
  };

  // Tạo mê cung trống mới
  const createEmptyMaze = (width, height) => {
    // Tạo mê cung trống với tường bao quanh
    const newMaze = Array(height).fill().map((_, row) => 
      Array(width).fill().map((_, col) => {
        if (row === 0 || col === 0 || row === height - 1 || col === width - 1) {
          return WALL;
        }
        return PATH;
      })
    );
    
    // Thiết lập điểm bắt đầu và kết thúc mặc định
    const newStartPos = [1, 1];
    const newGoalPos = [height - 2, width - 2];
    
    newMaze[newStartPos[0]][newStartPos[1]] = START;
    newMaze[newGoalPos[0]][newGoalPos[1]] = GOAL;
    
    setMaze(newMaze);
    setMazeSize({ width, height });
    setStartPos(newStartPos);
    setGoalPos(newGoalPos);
    
    return { maze: newMaze, startPos: newStartPos, goalPos: newGoalPos };
  };

  // Đặt lại đường đi và trạng thái giải
  const resetSolution = () => {
    setPath([]);
    setQValues(null);
    setIsSolving(false);
  };

  const contextValue = {
    // State
    maze, setMaze,
    mazeSize, setMazeSize,
    startPos, setStartPos,
    goalPos, setGoalPos,
    generatorAlgorithm, setGeneratorAlgorithm,
    solverAlgorithm, setSolverAlgorithm,
    generationSpeed, setGenerationSpeed,
    simulationSpeed, setSimulationSpeed,
    path, setPath,
    qValues, setQValues,
    isGenerating, setIsGenerating,
    isSolving, setIsSolving,
    showHeatmap, setShowHeatmap,
    
    // Methods
    saveMaze,
    getSavedMazes,
    loadMaze,
    createEmptyMaze,
    resetSolution
  };

  return (
    <MazeContext.Provider value={contextValue}>
      {children}
    </MazeContext.Provider>
  );
};

export default MazeContext;