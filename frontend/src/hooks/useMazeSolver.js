import { useState, useEffect, useCallback } from 'react';
import { useMazeContext } from '../context/MazeContext';
import mazeApi from '../api/mazeApi';

// Hook quản lý quá trình giải mê cung
const useMazeSolver = () => {
  const {
    maze,
    startPos,
    goalPos,
    solverAlgorithm,
    simulationSpeed,
    path, setPath,
    qValues, setQValues,
    isSolving, setIsSolving,
    resetSolution
  } = useMazeContext();

  // State cho quá trình mô phỏng
  const [currentPathIndex, setCurrentPathIndex] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [solutionStats, setSolutionStats] = useState(null);

  // Reset quá trình giải
  const resetSimulation = useCallback(() => {
    resetSolution();
    setCurrentPathIndex(0);
    setIsAnimating(false);
    setErrorMessage('');
    setSolutionStats(null);
  }, [resetSolution]);

  // Gửi mê cung đến API để giải
  const solveMaze = useCallback(async () => {
    if (!maze) {
      setErrorMessage('Cần tạo mê cung trước khi giải');
      return;
    }

    try {
      setIsSolving(true);
      setErrorMessage('');

      // Chuẩn bị dữ liệu để gửi
      const mazeData = {
        maze: maze,
        algorithm: solverAlgorithm,
        start_pos: startPos,
        goal_pos: goalPos
      };

      // Gọi API
      const result = await mazeApi.solveMaze(mazeData);

      // Xử lý kết quả
      setPath(result.path);
      setQValues(result.q_values);
      setSolutionStats({
        steps: result.steps,
        totalReward: result.total_reward
      });

      // Bắt đầu ở vị trí đầu tiên
      setCurrentPathIndex(0);

      return result;
    } catch (error) {
      console.error('Lỗi khi giải mê cung:', error);
      setErrorMessage(`Lỗi khi giải mê cung: ${error.message || 'Không xác định'}`);
      setIsSolving(false);
      return null;
    }
  }, [
    maze,
    solverAlgorithm,
    startPos,
    goalPos,
    setPath,
    setQValues,
    setIsSolving
  ]);

  // Mô phỏng quá trình giải mê cung
  useEffect(() => {
    if (isAnimating && path && currentPathIndex < path.length) {
      const timer = setTimeout(() => {
        if (currentPathIndex === path.length - 1) {
          setIsAnimating(false);
          setIsSolving(false);
        } else {
          setCurrentPathIndex(currentPathIndex + 1);
        }
      }, simulationSpeed);

      return () => clearTimeout(timer);
    }
  }, [
    isAnimating,
    currentPathIndex,
    path,
    simulationSpeed,
    setIsSolving
  ]);

  // Bắt đầu mô phỏng
  const startAnimation = useCallback(() => {
    if (path && path.length > 0) {
      setIsAnimating(true);
      setCurrentPathIndex(0);
    }
  }, [path]);

  // Dừng mô phỏng
  const stopAnimation = useCallback(() => {
    setIsAnimating(false);
  }, []);

  // Hiển thị kết quả cuối cùng
  const skipToEnd = useCallback(() => {
    if (path && path.length > 0) {
      setIsAnimating(false);
      setCurrentPathIndex(path.length - 1);
      setIsSolving(false);
    }
  }, [path, setIsSolving]);

  // Hiển thị từng bước
  const nextStep = useCallback(() => {
    if (path && currentPathIndex < path.length - 1) {
      setCurrentPathIndex(currentPathIndex + 1);
    } else {
      setIsSolving(false);
    }
  }, [currentPathIndex, path, setIsSolving]);

  const prevStep = useCallback(() => {
    if (currentPathIndex > 0) {
      setCurrentPathIndex(currentPathIndex - 1);
    }
  }, [currentPathIndex]);

  // Lấy vị trí hiện tại của chuột
  const getCurrentMousePosition = useCallback(() => {
    if (path && currentPathIndex < path.length) {
      return path[currentPathIndex];
    }
    return startPos;
  }, [path, currentPathIndex, startPos]);

  return {
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
    totalSteps: path ? path.length : 0,
    errorMessage,
    solutionStats
  };
};

export default useMazeSolver;