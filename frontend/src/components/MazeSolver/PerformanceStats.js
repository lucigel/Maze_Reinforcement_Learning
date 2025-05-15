import React from 'react';
import '../../styles/MazeSolver.css';

const PerformanceStats = ({ stats }) => {
  const { steps, totalReward } = stats;

  return (
    <div className="performance-stats">
      <h3>Thống kê hiệu suất</h3>
      
      <div className="stats-grid">
        <div className="stat-item">
          <div className="stat-label">Số bước:</div>
          <div className="stat-value">{steps}</div>
        </div>
        
        <div className="stat-item">
          <div className="stat-label">Tổng phần thưởng:</div>
          <div className="stat-value">{totalReward.toFixed(2)}</div>
        </div>
        
        <div className="stat-item">
          <div className="stat-label">Phần thưởng trung bình/bước:</div>
          <div className="stat-value">{(totalReward / steps).toFixed(2)}</div>
        </div>
      </div>
      
      <div className="stats-interpretation">
        <h4>Ý nghĩa:</h4>
        <ul>
          <li>
            <strong>Số bước:</strong> Số lượng hành động cần thiết để đi từ điểm bắt đầu đến đích.
          </li>
          <li>
            <strong>Tổng phần thưởng:</strong> Tổng của tất cả phần thưởng nhận được trong quá trình đi từ điểm bắt đầu đến đích.
          </li>
          <li>
            <strong>Phần thưởng trung bình/bước:</strong> Phần thưởng trung bình cho mỗi bước đi. Giá trị cao hơn thường cho thấy đường đi hiệu quả hơn.
          </li>
        </ul>
      </div>
    </div>
  );
};

export default PerformanceStats;