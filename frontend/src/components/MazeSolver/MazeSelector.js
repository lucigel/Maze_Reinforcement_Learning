import React from 'react';
import { useMazeContext } from '../../context/MazeContext';
import { toast } from 'react-toastify';
import '../../styles/MazeSolver.css';

const MazeSelector = () => {
  const { 
    loadMaze, 
    getSavedMazes, 
    createEmptyMaze, 
    resetSolution
  } = useMazeContext();

  // Lấy danh sách mê cung đã lưu
  const savedMazes = getSavedMazes();

  // Xử lý việc load mê cung đã lưu
  const handleLoadMaze = (index) => {
    const success = loadMaze(index);
    
    if (success) {
      resetSolution();
      toast.success('Đã tải mê cung thành công!');
    } else {
      toast.error('Không thể tải mê cung!');
    }
  };

  // Xử lý việc tạo mê cung mới
  const handleCreateMaze = (size) => {
    createEmptyMaze(size, size);
    resetSolution();
    toast.info(`Đã tạo mê cung trống ${size}x${size}`);
  };

  // Format timestamp thành chuỗi thời gian
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <div className="maze-selector">
      <h3>Chọn mê cung</h3>
      
      {savedMazes.length > 0 ? (
        <div className="saved-mazes-list">
          <p>Mê cung đã lưu:</p>
          <div className="maze-list">
            {savedMazes.map((maze, index) => (
              <div key={index} className="maze-item">
                <button 
                  className="btn btn-outline"
                  onClick={() => handleLoadMaze(index)}
                >
                  {maze.mazeSize.width}x{maze.mazeSize.height} 
                  ({maze.generatorAlgorithm.toUpperCase()})
                  <span className="maze-timestamp">
                    {formatTimestamp(maze.timestamp)}
                  </span>
                </button>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="no-mazes-message">
          Chưa có mê cung nào được lưu. Hãy quay lại trang Sinh mê cung để tạo và lưu mê cung.
        </p>
      )}
      
      <div className="create-empty-maze">
        <p>Hoặc tạo mê cung trống mới:</p>
        <div className="size-buttons">
          <button className="btn btn-secondary" onClick={() => handleCreateMaze(11)}>
            11x11
          </button>
          <button className="btn btn-secondary" onClick={() => handleCreateMaze(15)}>
            15x15
          </button>
          <button className="btn btn-secondary" onClick={() => handleCreateMaze(21)}>
            21x21
          </button>
        </div>
      </div>
    </div>
  );
};

export default MazeSelector;