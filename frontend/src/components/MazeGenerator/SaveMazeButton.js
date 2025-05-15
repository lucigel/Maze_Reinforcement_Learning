import React from 'react';
import { useMazeContext } from '../../context/MazeContext';
import { toast } from 'react-toastify';
import '../../styles/MazeGenerator.css';

const SaveMazeButton = () => {
  const { saveMaze } = useMazeContext();

  const handleSave = () => {
    const success = saveMaze();
    if (success) {
      toast.success('Mê cung đã được lưu thành công!');
    } else {
      toast.error('Không thể lưu mê cung. Hãy sinh mê cung trước.');
    }
  };

  return (
    <div className="save-maze-button">
      <button className="btn btn-success" onClick={handleSave}>
        <i className="fas fa-save"></i> Lưu mê cung này
      </button>
      <p className="hint">Mê cung sẽ được lưu để sử dụng ở trang Giải mê cung</p>
    </div>
  );
};

export default SaveMazeButton;