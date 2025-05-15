import React from 'react';
import '../../styles/MazeGenerator.css';

const GeneratorControls = ({
  width,
  height,
  algorithm,
  speed,
  onSizeChange,
  onApplySize,
  onAlgorithmChange,
  onSpeedChange,
  onGenerate,
  isGenerating
}) => {
  return (
    <div className="generator-controls">
      <div className="control-group">
        <h3>Kích thước mê cung</h3>
        <div className="size-controls">
          <div className="input-group">
            <label htmlFor="width">Chiều rộng:</label>
            <input
              type="number"
              id="width"
              name="width"
              value={width}
              onChange={onSizeChange}
              min="5"
              max="51"
              step="2"
              disabled={isGenerating}
            />
          </div>
          <div className="input-group">
            <label htmlFor="height">Chiều cao:</label>
            <input
              type="number"
              id="height"
              name="height"
              value={height}
              onChange={onSizeChange}
              min="5"
              max="51"
              step="2"
              disabled={isGenerating}
            />
          </div>
          <button 
            className="btn btn-secondary" 
            onClick={onApplySize}
            disabled={isGenerating}
          >
            Áp dụng
          </button>
        </div>
      </div>

      <div className="control-group">
        <h3>Thuật toán sinh mê cung</h3>
        <div className="algorithm-controls">
          <div className="radio-group">
            <input
              type="radio"
              id="dfs"
              name="algorithm"
              value="dfs"
              checked={algorithm === 'dfs'}
              onChange={onAlgorithmChange}
              disabled={isGenerating}
            />
            <label htmlFor="dfs">DFS (Thuật toán tìm kiếm sâu)</label>
          </div>
          <div className="radio-group">
            <input
              type="radio"
              id="prim"
              name="algorithm"
              value="prim"
              checked={algorithm === 'prim'}
              onChange={onAlgorithmChange}
              disabled={isGenerating}
            />
            <label htmlFor="prim">Prim (Thuật toán cây khung nhỏ nhất)</label>
          </div>
          <div className="radio-group">
            <input
              type="radio"
              id="wilson"
              name="algorithm"
              value="wilson"
              checked={algorithm === 'wilson'}
              onChange={onAlgorithmChange}
              disabled={isGenerating}
            />
            <label htmlFor="wilson">Wilson (Thuật toán đi bộ ngẫu nhiên)</label>
          </div>
        </div>
      </div>

      <div className="control-group">
        <h3>Tốc độ mô phỏng</h3>
        <div className="speed-controls">
          <input
            type="range"
            id="speed"
            name="speed"
            min="10"
            max="500"
            step="10"
            value={speed}
            onChange={onSpeedChange}
            disabled={isGenerating}
          />
          <div className="speed-labels">
            <span>Nhanh</span>
            <span>Chậm</span>
          </div>
        </div>
      </div>

      <button
        className="btn btn-primary generate-btn"
        onClick={onGenerate}
        disabled={isGenerating}
      >
        Sinh mê cung
      </button>
    </div>
  );
};

export default GeneratorControls;