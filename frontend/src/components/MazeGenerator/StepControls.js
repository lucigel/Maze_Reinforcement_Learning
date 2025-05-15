import React from 'react';
import '../../styles/MazeGenerator.css';

const StepControls = ({
  onPrev,
  onNext,
  onStart,
  onStop,
  onSkipToEnd,
  isAnimating,
  currentStep,
  totalSteps,
  disabled
}) => {
  return (
    <div className="step-controls">
      <h3>Điều khiển từng bước</h3>
      <div className="step-buttons">
        <button 
          className="btn btn-control" 
          onClick={onPrev} 
          disabled={disabled || currentStep <= 1 || isAnimating}
        >
          <i className="fas fa-step-backward"></i> Lùi lại
        </button>
        
        {isAnimating ? (
          <button 
            className="btn btn-control" 
            onClick={onStop} 
            disabled={disabled}
          >
            <i className="fas fa-pause"></i> Tạm dừng
          </button>
        ) : (
          <button 
            className="btn btn-control" 
            onClick={onStart} 
            disabled={disabled || currentStep >= totalSteps}
          >
            <i className="fas fa-play"></i> Tiếp tục
          </button>
        )}
        
        <button 
          className="btn btn-control" 
          onClick={onNext} 
          disabled={disabled || currentStep >= totalSteps || isAnimating}
        >
          <i className="fas fa-step-forward"></i> Tiến lên
        </button>
        
        <button 
          className="btn btn-control" 
          onClick={onSkipToEnd} 
          disabled={disabled || currentStep >= totalSteps}
        >
          <i className="fas fa-fast-forward"></i> Đến cuối
        </button>
      </div>
      <div className="step-progress">
        <span>Bước: {currentStep}/{totalSteps}</span>
        <progress value={currentStep} max={totalSteps}></progress>
      </div>
    </div>
  );
};

export default StepControls;