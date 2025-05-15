import React from 'react';
import '../../styles/MazeSolver.css';

const ModelSelector = ({ algorithm, onAlgorithmChange, speed, onSpeedChange, disabled }) => {
  return (
    <div className="model-selector">
      <h3>Lựa chọn Mô hình</h3>
      
      <div className="algorithm-options">
        <div className="radio-group">
          <input
            type="radio"
            id="q_learning"
            name="algorithm"
            value="q_learning"
            checked={algorithm === 'q_learning'}
            onChange={onAlgorithmChange}
            disabled={disabled}
          />
          <label htmlFor="q_learning">Q-Learning</label>
          <span className="algorithm-hint">
            Thuật toán off-policy, học từ trải nghiệm tốt nhất.
          </span>
        </div>
        
        <div className="radio-group">
          <input
            type="radio"
            id="sarsa"
            name="algorithm"
            value="sarsa"
            checked={algorithm === 'sarsa'}
            onChange={onAlgorithmChange}
            disabled={disabled}
          />
          <label htmlFor="sarsa">SARSA</label>
          <span className="algorithm-hint">
            Thuật toán on-policy, học từ trải nghiệm thực tế.
          </span>
        </div>
      </div>
      
      <div className="simulation-speed">
        <h3>Tốc độ mô phỏng</h3>
        <div className="speed-controls">
          <input
            type="range"
            id="simulation-speed"
            name="simulation-speed"
            min="50"
            max="500"
            step="10"
            value={speed}
            onChange={onSpeedChange}
            disabled={disabled}
          />
          <div className="speed-labels">
            <span>Nhanh</span>
            <span>Chậm</span>
          </div>
        </div>
      </div>
      
      <div className="model-description">
        <h4>Giới thiệu về {algorithm === 'q_learning' ? 'Q-Learning' : 'SARSA'}</h4>
        <p>
          {algorithm === 'q_learning' ? (
            <>
              Q-Learning là thuật toán học tăng cường "off-policy", học từ trải nghiệm có giá trị nhất, không phụ thuộc vào chính sách hiện tại. Nó sử dụng công thức:
              <br />
              <code>Q(s,a) = Q(s,a) + α[r + γmax<sub>a'</sub>Q(s',a') - Q(s,a)]</code>
            </>
          ) : (
            <>
              SARSA là thuật toán học tăng cường "on-policy", học từ trải nghiệm thực tế dựa trên chính sách hiện tại. Nó sử dụng công thức:
              <br />
              <code>Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]</code>
            </>
          )}
        </p>
      </div>
    </div>
  );
};

export default ModelSelector;