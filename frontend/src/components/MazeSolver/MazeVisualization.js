import React, { useRef, useEffect } from 'react';
import { useMazeContext, PATH, WALL, START, GOAL } from '../../context/MazeContext';
import '../../styles/MazeSolver.css';

const MazeVisualization = () => {
  const { 
    maze, 
    mazeSize, 
    path, 
    showHeatmap, 
    qValues
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
            // Vẽ chữ "S" ở điểm bắt đầu
            ctx.fillStyle = '#FFFFFF';
            ctx.font = `bold ${cellSize/2}px Arial`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('S', x + cellSize/2, y + cellSize/2);
            break;
          case GOAL:
            ctx.fillStyle = '#F44336';
            ctx.fillRect(x, y, cellSize, cellSize);
            // Vẽ ký hiệu phô mai ở đích
            ctx.fillStyle = '#FFEB3B';
            ctx.beginPath();
            ctx.arc(x + cellSize/2, y + cellSize/2, cellSize/3, 0, Math.PI * 2);
            ctx.fill();
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
    if (path && path.length > 0 && !showHeatmap) {
      ctx.strokeStyle = 'rgba(33, 150, 243, 0.7)';
      ctx.lineWidth = cellSize / 4;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      
      ctx.beginPath();
      
      for (let i = 0; i < path.length; i++) {
        const [row, col] = path[i];
        const x = col * cellSize + cellSize / 2;
        const y = row * cellSize + cellSize / 2;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.stroke();
    }

    // Vẽ heatmap nếu có và được kích hoạt
    if (showHeatmap && qValues && qValues.value_function) {
      const valueFunction = qValues.value_function;
      
      for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) {
          // Chỉ vẽ heatmap trên các ô đường đi
          if (maze[row][col] === PATH) {
            const value = valueFunction[row][col];
            
            // Chuẩn hóa giá trị để màu phù hợp
            // Giả sử giá trị từ -100 đến 100
            const normalizedValue = (value + 100) / 200;
            
            // Tạo màu gradient từ đỏ (thấp) -> vàng -> xanh (cao)
            let r, g, b;
            if (normalizedValue < 0.5) {
              // Từ đỏ đến vàng
              r = 255;
              g = Math.floor(normalizedValue * 2 * 255);
              b = 0;
            } else {
              // Từ vàng đến xanh
              r = Math.floor((1 - (normalizedValue - 0.5) * 2) * 255);
              g = 255;
              b = 0;
            }
            
            const color = `rgba(${r}, ${g}, ${b}, 0.7)`;
            
            const x = col * cellSize;
            const y = row * cellSize;
            
            // Vẽ màu nền cho ô
            ctx.fillStyle = color;
            ctx.fillRect(x, y, cellSize, cellSize);
            
            // Vẽ viền ô
            ctx.strokeStyle = '#CCCCCC';
            ctx.strokeRect(x, y, cellSize, cellSize);
            
            // Hiển thị giá trị
            ctx.fillStyle = '#000000';
            ctx.font = `${cellSize/4}px Arial`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(value.toFixed(1), x + cellSize/2, y + cellSize/2);
          }
        }
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
  }, [maze, path, showHeatmap, qValues, mazeSize]);

  return (
    <div className="maze-visualization">
      <canvas ref={canvasRef} className="maze-canvas"></canvas>
    </div>
  );
};

export default MazeVisualization;