import React, { useRef, useEffect } from 'react';
import { useMazeContext } from '../../context/MazeContext';
import '../../styles/MazeSolver.css';

const MouseVisualization = ({ position }) => {
  const { 
    maze, 
    mazeSize 
  } = useMazeContext();
  
  const canvasRef = useRef(null);

  // Hàm vẽ chuột
  const drawMouse = () => {
    const canvas = canvasRef.current;
    if (!canvas || !maze || !position) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { width, height } = mazeSize;
    const cellSize = Math.min(
      Math.floor(canvas.width / width),
      Math.floor(canvas.height / height)
    );

    const [row, col] = position;
    const x = col * cellSize + cellSize / 2;
    const y = row * cellSize + cellSize / 2;
    
    // Vẽ thân chuột
    ctx.fillStyle = '#2196F3';
    ctx.beginPath();
    ctx.arc(x, y, cellSize / 2, 0, Math.PI * 2);
    ctx.fill();
    
    // Vẽ tai chuột
    ctx.fillStyle = '#1976D2';
    ctx.beginPath();
    ctx.arc(x - cellSize / 3, y - cellSize / 3, cellSize / 6, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(x + cellSize / 3, y - cellSize / 3, cellSize / 6, 0, Math.PI * 2);
    ctx.fill();
    
    // Vẽ mắt chuột
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(x - cellSize / 6, y - cellSize / 8, cellSize / 8, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(x + cellSize / 6, y - cellSize / 8, cellSize / 8, 0, Math.PI * 2);
    ctx.fill();
    
    // Vẽ đồng tử
    ctx.fillStyle = '#000000';
    ctx.beginPath();
    ctx.arc(x - cellSize / 6, y - cellSize / 8, cellSize / 16, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(x + cellSize / 6, y - cellSize / 8, cellSize / 16, 0, Math.PI * 2);
    ctx.fill();
    
    // Vẽ mũi chuột
    ctx.fillStyle = '#FF4081';
    ctx.beginPath();
    ctx.arc(x, y + cellSize / 8, cellSize / 10, 0, Math.PI * 2);
    ctx.fill();
    
    // Vẽ ria chuột
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = cellSize / 20;
    
    // Ria trái
    ctx.beginPath();
    ctx.moveTo(x - cellSize / 10, y + cellSize / 12);
    ctx.lineTo(x - cellSize / 2, y);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(x - cellSize / 10, y + cellSize / 6);
    ctx.lineTo(x - cellSize / 2, y + cellSize / 4);
    ctx.stroke();
    
    // Ria phải
    ctx.beginPath();
    ctx.moveTo(x + cellSize / 10, y + cellSize / 12);
    ctx.lineTo(x + cellSize / 2, y);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(x + cellSize / 10, y + cellSize / 6);
    ctx.lineTo(x + cellSize / 2, y + cellSize / 4);
    ctx.stroke();
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
      drawMouse();
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

  // Vẽ lại chuột khi có thay đổi
  useEffect(() => {
    drawMouse();
  }, [maze, position, mazeSize]);

  return (
    <canvas ref={canvasRef} className="mouse-canvas"></canvas>
  );
};

export default MouseVisualization;