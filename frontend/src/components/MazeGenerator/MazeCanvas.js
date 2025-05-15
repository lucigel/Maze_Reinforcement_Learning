import React, { useRef, useEffect } from 'react';
import { useMazeContext, PATH, WALL, START, GOAL } from '../../context/MazeContext';
import '../../styles/MazeGenerator.css';

const MazeCanvas = () => {
  const { 
    maze, 
    mazeSize, 
    path, 
    currentPathIndex = 0 
  } = useMazeContext();
  
  const canvasRef = useRef(null);

  // Hàm vẽ mê cung
  const drawMaze = () => {
    const canvas = canvasRef.current;
    if (!canvas || !maze) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { width, height } = mazeSize;
    const cellSize = Math.min(
      Math.floor(canvas.width / width),
      Math.floor(canvas.height / height)
    );

    // Vẽ mê cung
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        const x = col * cellSize;
        const y = row * cellSize;
        
        switch (maze[row][col]) {
          case WALL:
            ctx.fillStyle = '#333333';
            ctx.fillRect(x, y, cellSize, cellSize);
            break;
          case PATH:
            ctx.fillStyle = '#FFFFFF';
            ctx.fillRect(x, y, cellSize, cellSize);
            break;
          case START:
            ctx.fillStyle = '#4CAF50';
            ctx.fillRect(x, y, cellSize, cellSize);
            break;
          case GOAL:
            ctx.fillStyle = '#F44336';
            ctx.fillRect(x, y, cellSize, cellSize);
            break;
          default:
            ctx.fillStyle = '#FFFFFF';
            ctx.fillRect(x, y, cellSize, cellSize);
        }
        
        // Vẽ viền ô
        ctx.strokeStyle = '#CCCCCC';
        ctx.strokeRect(x, y, cellSize, cellSize);
      }
    }

    // Vẽ đường đi nếu có
    if (path && path.length > 0) {
      for (let i = 0; i <= currentPathIndex && i < path.length; i++) {
        const [row, col] = path[i];
        
        // Bỏ qua điểm bắt đầu và kết thúc
        if (maze[row][col] === START || maze[row][col] === GOAL) continue;
        
        const x = col * cellSize + cellSize / 2;
        const y = row * cellSize + cellSize / 2;
        
        ctx.fillStyle = 'rgba(33, 150, 243, 0.5)';
        ctx.beginPath();
        ctx.arc(x, y, cellSize / 3, 0, Math.PI * 2);
        ctx.fill();
      }
      
      // Vẽ vị trí hiện tại của chuột nếu đang mô phỏng
      if (currentPathIndex < path.length) {
        const [row, col] = path[currentPathIndex];
        const x = col * cellSize + cellSize / 2;
        const y = row * cellSize + cellSize / 2;
        
        ctx.fillStyle = '#2196F3';
        ctx.beginPath();
        ctx.arc(x, y, cellSize / 2.5, 0, Math.PI * 2);
        ctx.fill();
        
        // Vẽ mắt cho chuột
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(x - cellSize / 8, y - cellSize / 8, cellSize / 10, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(x + cellSize / 8, y - cellSize / 8, cellSize / 10, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };

  // Điều chỉnh kích thước canvas khi component được mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      const container = canvas.parentElement;
      const size = Math.min(container.clientWidth, container.clientHeight);
      canvas.width = size;
      canvas.height = size;
      drawMaze();
    };

    // Điều chỉnh kích thước ban đầu
    resizeCanvas();

    // Thêm event listener cho việc thay đổi kích thước
    window.addEventListener('resize', resizeCanvas);

    // Cleanup
    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  // Vẽ lại mê cung khi có thay đổi
  useEffect(() => {
    drawMaze();
  }, [maze, path, currentPathIndex, mazeSize]);

  return (
    <div className="maze-canvas-container">
      <canvas ref={canvasRef} className="maze-canvas"></canvas>
    </div>
  );
};

export default MazeCanvas;