import React, { useRef, useEffect } from 'react';
import { useMazeContext, PATH, WALL, START, GOAL } from '../../context/MazeContext';
import '../../styles/MazeSolver.css';

const ValueFunctionHeatmap = ({ qValues, maze }) => {
  const { mazeSize } = useMazeContext();
  const canvasRef = useRef(null);

  // Hàm vẽ heatmap
  const drawHeatmap = () => {
    const canvas = canvasRef.current;
    if (!canvas || !qValues || !maze) return;

    const valueFunction = qValues.value_function;
    if (!valueFunction) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { width, height } = mazeSize;
    const cellSize = Math.min(
      Math.floor(canvas.width / width),
      Math.floor(canvas.height / height)
    );

    // Tìm giá trị lớn nhất và nhỏ nhất để chuẩn hóa
    let minValue = Infinity;
    let maxValue = -Infinity;

    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        if (maze[row][col] !== WALL) {
          const value = valueFunction[row][col];
          minValue = Math.min(minValue, value);
          maxValue = Math.max(maxValue, value);
        }
      }
    }

    // Điều chỉnh phạm vi giá trị nếu quá nhỏ
    const valueRange = maxValue - minValue;
    if (valueRange < 1e-6) {
      minValue = -1;
      maxValue = 1;
    }

    // Vẽ heatmap
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        if (maze[row][col] === WALL) {
          // Vẽ tường
          ctx.fillStyle = '#333333';
        } else {
          // Vẽ ô với màu tương ứng với giá trị
          const value = valueFunction[row][col];
          const normalizedValue = (value - minValue) / (maxValue - minValue);
          
          // Tạo màu gradient từ xanh dương (thấp) -> vàng -> đỏ (cao)
          let color;
          if (normalizedValue < 0.5) {
            // Xanh dương đến vàng
            const r = Math.floor(normalizedValue * 2 * 255);
            const g = Math.floor(normalizedValue * 2 * 255);
            const b = Math.floor((1 - normalizedValue * 2) * 255);
            color = `rgb(${r}, ${g}, ${b})`;
          } else {
            // Vàng đến đỏ
            const r = 255;
            const g = Math.floor((1 - (normalizedValue - 0.5) * 2) * 255);
            const b = 0;
            color = `rgb(${r}, ${g}, ${b})`;
          }
          
          ctx.fillStyle = color;
        }
        
        const x = col * cellSize;
        const y = row * cellSize;
        ctx.fillRect(x, y, cellSize, cellSize);
        
        // Vẽ viền ô
        ctx.strokeStyle = '#666666';
        ctx.strokeRect(x, y, cellSize, cellSize);
        
        // Hiển thị giá trị cho các ô không phải tường
        if (maze[row][col] !== WALL) {
          const value = valueFunction[row][col];
          
          // Điều chỉnh màu chữ dựa trên màu nền
          ctx.fillStyle = '#000000';
          
          ctx.font = `${Math.floor(cellSize / 4)}px Arial`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(value.toFixed(1), x + cellSize / 2, y + cellSize / 2);
        }
      }
    }

    // Vẽ điểm bắt đầu và kết thúc
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        const x = col * cellSize;
        const y = row * cellSize;
        
        if (maze[row][col] === START) {
          // Vẽ viền cho điểm bắt đầu
          ctx.strokeStyle = '#4CAF50';
          ctx.lineWidth = 3;
          ctx.strokeRect(x + 2, y + 2, cellSize - 4, cellSize - 4);
          
          // Vẽ chữ "S"
          ctx.fillStyle = '#4CAF50';
          ctx.font = `bold ${Math.floor(cellSize / 2)}px Arial`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('S', x + cellSize / 2, y + cellSize / 2);
        } else if (maze[row][col] === GOAL) {
          // Vẽ viền cho điểm kết thúc
          ctx.strokeStyle = '#F44336';
          ctx.lineWidth = 3;
          ctx.strokeRect(x + 2, y + 2, cellSize - 4, cellSize - 4);
          
          // Vẽ chữ "G"
          ctx.fillStyle = '#F44336';
          ctx.font = `bold ${Math.floor(cellSize / 2)}px Arial`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('G', x + cellSize / 2, y + cellSize / 2);
        }
      }
    }

    // Vẽ thanh màu gradient bên dưới để tham chiếu
    const legendHeight = 20;
    const legendWidth = canvas.width - 40;
    const legendX = 20;
    const legendY = canvas.height - legendHeight - 30;
    
    // Vẽ gradient
    const gradient = ctx.createLinearGradient(legendX, 0, legendX + legendWidth, 0);
    gradient.addColorStop(0, 'blue');    // Giá trị thấp nhất
    gradient.addColorStop(0.5, 'yellow'); // Giá trị trung bình
    gradient.addColorStop(1, 'red');     // Giá trị cao nhất
    
    ctx.fillStyle = gradient;
    ctx.fillRect(legendX, legendY, legendWidth, legendHeight);
    
    // Vẽ viền cho thanh gradient
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);
    
    // Vẽ các nhãn
    ctx.fillStyle = '#000000';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    
    ctx.fillText(minValue.toFixed(1), legendX, legendY + legendHeight + 5);
    ctx.fillText(((maxValue + minValue) / 2).toFixed(1), legendX + legendWidth / 2, legendY + legendHeight + 5);
    ctx.fillText(maxValue.toFixed(1), legendX + legendWidth, legendY + legendHeight + 5);
    
    // Tiêu đề heatmap
    ctx.fillStyle = '#000000';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('Heatmap giá trị Q (Hàm giá trị)', canvas.width / 2, legendY - 10);
  };

  // Điều chỉnh kích thước canvas khi component được mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      const container = canvas.parentElement;
      const width = container.clientWidth;
      const height = Math.min(300, container.clientHeight / 2);
      canvas.width = width;
      canvas.height = height;
      drawHeatmap();
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

  // Vẽ lại heatmap khi có thay đổi
  useEffect(() => {
    drawHeatmap();
  }, [qValues, maze, mazeSize]);

  return (
    <div className="heatmap-container">
      <canvas ref={canvasRef} className="heatmap-canvas"></canvas>
    </div>
  );
};

export default ValueFunctionHeatmap;