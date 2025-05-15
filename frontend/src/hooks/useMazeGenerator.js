import { useState, useEffect, useCallback } from 'react';
import { useMazeContext, PATH, WALL, START, GOAL } from '../context/MazeContext';

// Hook quản lý quá trình sinh mê cung
const useMazeGenerator = () => {
  const { 
    maze, setMaze,
    mazeSize, 
    startPos, setStartPos,
    goalPos, setGoalPos,
    generatorAlgorithm,
    generationSpeed,
    isGenerating, setIsGenerating,
    createEmptyMaze
  } = useMazeContext();

  // State cho quá trình sinh mê cung
  const [generationSteps, setGenerationSteps] = useState([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [isAnimating, setIsAnimating] = useState(false);

  // Reset quá trình sinh
  const resetGeneration = useCallback(() => {
    setGenerationSteps([]);
    setCurrentStepIndex(-1);
    setIsAnimating(false);
    setIsGenerating(false);
  }, [setIsGenerating]);

  // Thuật toán DFS để sinh mê cung
  const generateMazeDFS = useCallback(() => {
    const { width, height } = mazeSize;
    
    // Tạo mê cung ban đầu toàn tường
    const initialMaze = Array(height).fill().map(() => Array(width).fill(WALL));
    
    // Các bước sinh mê cung
    const steps = [];
    
    // Hàm kiểm tra ô có hợp lệ không
    const isValid = (row, col) => {
      return row >= 0 && row < height && col >= 0 && col < width;
    };
    
    // Hàm kiểm tra ô có thể đi đến không (chưa thăm)
    const canVisit = (maze, row, col) => {
      if (!isValid(row, col)) return false;
      return maze[row][col] === WALL;
    };
    
    // Các hướng di chuyển: [row, col]
    const directions = [
      [-2, 0],  // Lên
      [2, 0],   // Xuống
      [0, -2],  // Trái
      [0, 2]    // Phải
    ];
    
    // Tạo đường đi bắt đầu
    const startRow = 1;
    const startCol = 1;
    initialMaze[startRow][startCol] = PATH;
    
    // Thêm bước đầu tiên
    steps.push(JSON.parse(JSON.stringify(initialMaze)));
    
    // Stack để theo dõi đường đi
    const stack = [[startRow, startCol]];
    
    // DFS
    while (stack.length > 0) {
      const [currentRow, currentCol] = stack[stack.length - 1];
      
      // Trộn ngẫu nhiên các hướng
      const shuffledDirections = [...directions].sort(() => Math.random() - 0.5);
      
      // Kiểm tra xem có thể di chuyển đến ô nào không
      let canMove = false;
      
      for (const [dr, dc] of shuffledDirections) {
        const newRow = currentRow + dr;
        const newCol = currentCol + dc;
        
        if (canVisit(initialMaze, newRow, newCol)) {
          // Đánh dấu ô trung gian
          initialMaze[currentRow + dr/2][currentCol + dc/2] = PATH;
          
          // Đánh dấu ô mới
          initialMaze[newRow][newCol] = PATH;
          
          // Thêm bước hiện tại
          steps.push(JSON.parse(JSON.stringify(initialMaze)));
          
          // Đưa ô mới vào stack
          stack.push([newRow, newCol]);
          
          canMove = true;
          break;
        }
      }
      
      // Nếu không thể di chuyển, lùi lại
      if (!canMove) {
        stack.pop();
      }
    }
    
    // Thiết lập điểm bắt đầu và kết thúc
    const newStartPos = [1, 1];
    const newGoalPos = [height - 2, width - 2];
    
    // Đảm bảo đích là đường đi
    initialMaze[newGoalPos[0]][newGoalPos[1]] = PATH;
    
    // Đánh dấu điểm bắt đầu và kết thúc
    initialMaze[newStartPos[0]][newStartPos[1]] = START;
    initialMaze[newGoalPos[0]][newGoalPos[1]] = GOAL;
    
    // Thêm bước cuối cùng
    steps.push(JSON.parse(JSON.stringify(initialMaze)));
    
    return { 
      steps, 
      startPos: newStartPos, 
      goalPos: newGoalPos 
    };
  }, [mazeSize]);

  // Thuật toán Prim để sinh mê cung
  const generateMazePrim = useCallback(() => {
    const { width, height } = mazeSize;
    
    // Tạo mê cung ban đầu toàn tường
    const initialMaze = Array(height).fill().map(() => Array(width).fill(WALL));
    
    // Các bước sinh mê cung
    const steps = [];
    
    // Hàm kiểm tra ô có hợp lệ không
    const isValid = (row, col) => {
      return row >= 0 && row < height && col >= 0 && col < width;
    };
    
    // Các hướng di chuyển: [row, col]
    const directions = [
      [-2, 0],  // Lên
      [2, 0],   // Xuống
      [0, -2],  // Trái
      [0, 2]    // Phải
    ];
    
    // Tạo ô bắt đầu
    const startRow = 1;
    const startCol = 1;
    initialMaze[startRow][startCol] = PATH;
    
    // Thêm bước đầu tiên
    steps.push(JSON.parse(JSON.stringify(initialMaze)));
    
    // Danh sách tường (walls) để xem xét
    let walls = [];
    
    // Thêm các tường xung quanh ô bắt đầu
    for (const [dr, dc] of directions) {
      const newRow = startRow + dr;
      const newCol = startCol + dc;
      
      if (isValid(newRow, newCol)) {
        walls.push([startRow + dr/2, startCol + dc/2, newRow, newCol]);
      }
    }
    
    // Prim's algorithm
    while (walls.length > 0) {
      // Chọn ngẫu nhiên một tường
      const wallIndex = Math.floor(Math.random() * walls.length);
      const [wallRow, wallCol, cellRow, cellCol] = walls[wallIndex];
      walls.splice(wallIndex, 1);
      
      // Nếu chỉ một trong hai ô bên cạnh tường là đường đi
      if (initialMaze[cellRow][cellCol] === WALL) {
        // Tạo đường đi qua tường
        initialMaze[wallRow][wallCol] = PATH;
        initialMaze[cellRow][cellCol] = PATH;
        
        // Thêm bước hiện tại
        steps.push(JSON.parse(JSON.stringify(initialMaze)));
        
        // Thêm tường mới xung quanh ô vừa mở
        for (const [dr, dc] of directions) {
          const newRow = cellRow + dr;
          const newCol = cellCol + dc;
          
          if (isValid(newRow, newCol) && initialMaze[newRow][newCol] === WALL) {
            walls.push([cellRow + dr/2, cellCol + dc/2, newRow, newCol]);
          }
        }
      }
    }
    
    // Thiết lập điểm bắt đầu và kết thúc
    const newStartPos = [1, 1];
    const newGoalPos = [height - 2, width - 2];
    
    // Đảm bảo đích là đường đi
    initialMaze[newGoalPos[0]][newGoalPos[1]] = PATH;
    
    // Đánh dấu điểm bắt đầu và kết thúc
    initialMaze[newStartPos[0]][newStartPos[1]] = START;
    initialMaze[newGoalPos[0]][newGoalPos[1]] = GOAL;
    
    // Thêm bước cuối cùng
    steps.push(JSON.parse(JSON.stringify(initialMaze)));
    
    return { 
      steps, 
      startPos: newStartPos, 
      goalPos: newGoalPos 
    };
  }, [mazeSize]);

  // Thuật toán Wilson để sinh mê cung
  const generateMazeWilson = useCallback(() => {
    const { width, height } = mazeSize;
    
    // Tạo mê cung ban đầu toàn tường
    const initialMaze = Array(height).fill().map(() => Array(width).fill(WALL));
    
    // Các bước sinh mê cung
    const steps = [];
    
    // Hàm kiểm tra ô có hợp lệ không
    const isValid = (row, col) => {
      return row >= 0 && row < height && col >= 0 && col < width && row % 2 === 1 && col % 2 === 1;
    };
    
    // Các hướng di chuyển: [row, col]
    const directions = [
      [-2, 0],  // Lên
      [2, 0],   // Xuống
      [0, -2],  // Trái
      [0, 2]    // Phải
    ];
    
    // Tạo danh sách tất cả các ô có thể thăm
    const allCells = [];
    for (let row = 1; row < height; row += 2) {
      for (let col = 1; col < width; col += 2) {
        allCells.push([row, col]);
      }
    }
    
    // Trộn ngẫu nhiên
    allCells.sort(() => Math.random() - 0.5);
    
    // Chọn ô bắt đầu
    const startCell = allCells.pop();
    const [startRow, startCol] = startCell;
    initialMaze[startRow][startCol] = PATH;
    
    // Thêm bước đầu tiên
    steps.push(JSON.parse(JSON.stringify(initialMaze)));
    
    // Danh sách các ô đã thăm
    const visitedCells = new Set();
    visitedCells.add(`${startRow},${startCol}`);
    
    // Wilson's algorithm
    while (allCells.length > 0) {
      // Chọn ô bắt đầu cho loop-erased random walk
      const [currentRow, currentCol] = allCells.pop();
      
      // Đường đi hiện tại
      let path = [[currentRow, currentCol]];
      let [walkRow, walkCol] = [currentRow, currentCol];
      
      // Random walk cho đến khi gặp ô đã thăm
      while (!visitedCells.has(`${walkRow},${walkCol}`)) {
        // Chọn hướng ngẫu nhiên
        const [dr, dc] = directions[Math.floor(Math.random() * directions.length)];
        
        // Ô tiếp theo
        let nextRow = walkRow + dr;
        let nextCol = walkCol + dc;
        
        // Đảm bảo ô hợp lệ, nếu không thì thử lại
        if (!isValid(nextRow, nextCol)) {
          continue;
        }
        
        // Kiểm tra xem có tạo vòng lặp không
        const pathIndex = path.findIndex(([r, c]) => r === nextRow && c === nextCol);
        
        if (pathIndex !== -1) {
          // Nếu có vòng lặp, cắt bỏ phần đó
          path = path.slice(0, pathIndex + 1);
        } else {
          // Nếu không, thêm ô mới vào đường đi
          path.push([nextRow, nextCol]);
        }
        
        // Cập nhật ô hiện tại
        [walkRow, walkCol] = [nextRow, nextCol];
      }
      
      // Thêm đường đi vào mê cung
      for (let i = 0; i < path.length - 1; i++) {
        const [r1, c1] = path[i];
        const [r2, c2] = path[i + 1];
        
        // Tạo đường đi
        initialMaze[r1][c1] = PATH;
        initialMaze[(r1 + r2) / 2][(c1 + c2) / 2] = PATH;
        initialMaze[r2][c2] = PATH;
        
        // Thêm bước hiện tại
        steps.push(JSON.parse(JSON.stringify(initialMaze)));
        
        // Đánh dấu các ô đã thăm
        visitedCells.add(`${r1},${c1}`);
      }
      
      // Đánh dấu ô cuối cùng của đường đi đã thăm
      const [lastRow, lastCol] = path[path.length - 1];
      visitedCells.add(`${lastRow},${lastCol}`);
    }
    
    // Thiết lập điểm bắt đầu và kết thúc
    const newStartPos = [1, 1];
    const newGoalPos = [height - 2, width - 2];
    
    // Đảm bảo đích là đường đi
    initialMaze[newGoalPos[0]][newGoalPos[1]] = PATH;
    
    // Đánh dấu điểm bắt đầu và kết thúc
    initialMaze[newStartPos[0]][newStartPos[1]] = START;
    initialMaze[newGoalPos[0]][newGoalPos[1]] = GOAL;
    
    // Thêm bước cuối cùng
    steps.push(JSON.parse(JSON.stringify(initialMaze)));
    
    return { 
      steps, 
      startPos: newStartPos, 
      goalPos: newGoalPos 
    };
  }, [mazeSize]);

  // Sinh mê cung dựa trên thuật toán đã chọn
  const generateMaze = useCallback(() => {
    try {
      setIsGenerating(true);
      let result;
      
      switch (generatorAlgorithm) {
        case 'dfs':
          result = generateMazeDFS();
          break;
        case 'prim':
          result = generateMazePrim();
          break;
        case 'wilson':
          result = generateMazeWilson();
          break;
        default:
          result = generateMazeDFS();
      }
      
      setGenerationSteps(result.steps);
      setStartPos(result.startPos);
      setGoalPos(result.goalPos);
      setCurrentStepIndex(0);
      
      return result;
    } catch (error) {
      console.error('Lỗi khi sinh mê cung:', error);
      setIsGenerating(false);
      return null;
    }
  }, [
    generatorAlgorithm, 
    generateMazeDFS, 
    generateMazePrim, 
    generateMazeWilson, 
    setIsGenerating, 
    setStartPos, 
    setGoalPos
  ]);

  // Mô phỏng quá trình sinh mê cung
  useEffect(() => {
    if (isAnimating && currentStepIndex >= 0 && currentStepIndex < generationSteps.length) {
      const timer = setTimeout(() => {
        setMaze(generationSteps[currentStepIndex]);
        
        if (currentStepIndex === generationSteps.length - 1) {
          setIsAnimating(false);
          setIsGenerating(false);
        } else {
          setCurrentStepIndex(currentStepIndex + 1);
        }
      }, generationSpeed);
      
      return () => clearTimeout(timer);
    }
  }, [
    isAnimating, 
    currentStepIndex, 
    generationSteps, 
    generationSpeed, 
    setMaze, 
    setIsGenerating
  ]);

  // Bắt đầu mô phỏng
  const startAnimation = useCallback(() => {
    if (generationSteps.length > 0) {
      setIsAnimating(true);
      setCurrentStepIndex(0);
    }
  }, [generationSteps]);

  // Dừng mô phỏng
  const stopAnimation = useCallback(() => {
    setIsAnimating(false);
  }, []);

  // Hiển thị kết quả cuối cùng
  const skipToEnd = useCallback(() => {
    if (generationSteps.length > 0) {
      setIsAnimating(false);
      setMaze(generationSteps[generationSteps.length - 1]);
      setCurrentStepIndex(generationSteps.length - 1);
      setIsGenerating(false);
    }
  }, [generationSteps, setMaze, setIsGenerating]);

  // Hiển thị từng bước
  const nextStep = useCallback(() => {
    if (currentStepIndex < generationSteps.length - 1) {
      setCurrentStepIndex(currentStepIndex + 1);
      setMaze(generationSteps[currentStepIndex + 1]);
    } else {
      setIsGenerating(false);
    }
  }, [currentStepIndex, generationSteps, setMaze, setIsGenerating]);

  const prevStep = useCallback(() => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(currentStepIndex - 1);
      setMaze(generationSteps[currentStepIndex - 1]);
    }
  }, [currentStepIndex, generationSteps, setMaze]);

  return {
    generateMaze,
    resetGeneration,
    startAnimation,
    stopAnimation,
    skipToEnd,
    nextStep,
    prevStep,
    isAnimating,
    currentStepIndex,
    totalSteps: generationSteps.length
  };
};

export default useMazeGenerator;