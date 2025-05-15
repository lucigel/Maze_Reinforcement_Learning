import axios from 'axios';

// Base URL của API
const API_BASE_URL = 'http://localhost:8000';

// Tạo instance Axios với config mặc định
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API service
const mazeApi = {
  // Gửi mê cung để giải
  solveMaze: async (mazeData) => {
    try {
      const response = await apiClient.post('/api/solve-maze', mazeData);
      return response.data;
    } catch (error) {
      console.error('Lỗi khi gọi API solve-maze:', error);
      throw error;
    }
  },

  // Lấy thông tin về các mô hình đã huấn luyện
  getModelsInfo: async () => {
    try {
      const response = await apiClient.get('/api/models-info');
      return response.data;
    } catch (error) {
      console.error('Lỗi khi gọi API models-info:', error);
      throw error;
    }
  },

  // Kiểm tra kết nối tới API
  checkHealth: async () => {
    try {
      const response = await apiClient.get('/health');
      return response.data.status === 'ok';
    } catch (error) {
      console.error('Lỗi khi kiểm tra sức khỏe API:', error);
      return false;
    }
  }
};

export default mazeApi;