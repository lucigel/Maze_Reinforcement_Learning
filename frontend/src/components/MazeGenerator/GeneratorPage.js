import React, { useState, useEffect } from 'react';
import { useMazeContext } from '../../context/MazeContext';
import useMazeGenerator from '../../hooks/useMazeGenerator';
import GeneratorControls from './GeneratorControls';
import MazeCanvas from './MazeCanvas';
import StepControls from './StepControls';
import SaveMazeButton from './SaveMazeButton';
import { toast } from 'react-toastify';
import '../../styles/MazeGenerator.css';

const GeneratorPage = () => {
  const { 
    mazeSize, setMazeSize,
    generatorAlgorithm, setGeneratorAlgorithm,
    generationSpeed, setGenerationSpeed,
    isGenerating, maze
  } = useMazeContext();

  const [sizeInput, setSizeInput] = useState({
    width: mazeSize.width,
    height: mazeSize.height
  });

  const {
    generateMaze,
    resetGeneration,
    startAnimation,
    stopAnimation,
    skipToEnd,
    nextStep,
    prevStep,
    isAnimating,
    currentStepIndex,
    totalSteps
  } = useMazeGenerator();

  // Xử lý thay đổi kích thước
  const handleSizeChange = (e) => {
    const { name, value } = e.target;
    const parsedValue = parseInt(value, 10);
    
    // Kiểm tra giá trị hợp lệ
    if (!isNaN(parsedValue) && parsedValue >= 5 && parsedValue <= 51) {
      setSizeInput({
        ...sizeInput,
        [name]: parsedValue
      });
    }
  };

  // Áp dụng kích thước mới
  const applySize = () => {
    // Đảm bảo kích thước lẻ để thuật toán hoạt động tốt
    const width = sizeInput.width % 2 === 0 ? sizeInput.width + 1 : sizeInput.width;
    const height = sizeInput.height % 2 === 0 ? sizeInput.height + 1 : sizeInput.height;
    
    setMazeSize({ width, height });
    resetGeneration();
    toast.info(`Đã thay đổi kích thước mê cung thành ${width}x${height}`);
  };

  // Xử lý thay đổi thuật toán
  const handleAlgorithmChange = (e) => {
    setGeneratorAlgorithm(e.target.value);
    resetGeneration();
  };

  // Xử lý thay đổi tốc độ sinh
  const handleSpeedChange = (e) => {
    setGenerationSpeed(parseInt(e.target.value, 10));
  };

  // Xử lý việc bắt đầu sinh mê cung
  const handleGenerateMaze = () => {
    resetGeneration();
    generateMaze();
    startAnimation();
  };

  // Hiển thị trạng thái sinh mê cung
  const renderStatus = () => {
    if (isGenerating) {
      return `Đang sinh mê cung: Bước ${currentStepIndex + 1}/${totalSteps}`;
    }
    return maze ? 'Mê cung đã được tạo' : 'Chưa có mê cung';
  };

  return (
    <div className="generator-page">
      <h2>Sinh mê cung</h2>
      
      <div className="generator-layout">
        <div className="control-panel">
          <GeneratorControls 
            width={sizeInput.width}
            height={sizeInput.height}
            algorithm={generatorAlgorithm}
            speed={generationSpeed}
            onSizeChange={handleSizeChange}
            onApplySize={applySize}
            onAlgorithmChange={handleAlgorithmChange}
            onSpeedChange={handleSpeedChange}
            onGenerate={handleGenerateMaze}
            isGenerating={isGenerating}
          />
          
          {isGenerating && (
            <StepControls
              onNext={nextStep}
              onPrev={prevStep}
              onSkipToEnd={skipToEnd}
              onStop={stopAnimation}
              onStart={startAnimation}
              isAnimating={isAnimating}
              currentStep={currentStepIndex + 1}
              totalSteps={totalSteps}
              disabled={!isGenerating}
            />
          )}
          
          {maze && !isGenerating && (
            <SaveMazeButton />
          )}
          
          <div className="status-display">
            <p>{renderStatus()}</p>
          </div>
        </div>
        
        <div className="maze-display">
          <MazeCanvas />
        </div>
      </div>
    </div>
  );
};

export default GeneratorPage;