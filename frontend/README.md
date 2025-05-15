# Mô phỏng mê cung và Học tăng cường - Frontend

Dự án này mô phỏng quá trình sinh mê cung bằng các thuật toán khác nhau và giải mê cung bằng các thuật toán học tăng cường.

## Cài đặt

1. Cài đặt các gói phụ thuộc:

```bash
npm install
```

2. Chạy ứng dụng ở chế độ phát triển:

```bash
npm start
```

Ứng dụng sẽ mở tại [http://localhost:3000](http://localhost:3000).

## Cấu trúc dự án

```
src/
├── api/
│   └── mazeApi.js              # Gọi API đến backend
│
├── components/
│   ├── Common/                 # Components chung
│   │   ├── Header.js           # Header trang web
│   │   ├── Footer.js           # Footer trang web
│   │   └── Navigation.js       # Menu điều hướng
│   │
│   ├── MazeGenerator/          # Components cho trang 1
│   │   ├── GeneratorPage.js    # Trang sinh mê cung
│   │   ├── GeneratorControls.js # Các nút điều khiển sinh mê cung
│   │   ├── MazeCanvas.js       # Canvas hiển thị mê cung
│   │   ├── StepControls.js     # Điều khiển từng bước sinh
│   │   └── SaveMazeButton.js   # Nút lưu mê cung
│   │
│   └── MazeSolver/             # Components cho trang 2
│       ├── SolverPage.js       # Trang mô phỏng chuột
│       ├── ModelSelector.js    # Chọn model học tăng cường
│       ├── MazeSelector.js     # Chọn mê cung đã lưu
│       ├── MazeVisualization.js # Hiển thị mê cung
│       ├── MouseVisualization.js # Hiển thị chuột
│       ├── ValueFunctionHeatmap.js # Hiển thị heatmap giá trị Q
│       └── PerformanceStats.js # Thống kê hiệu suất
│
├── context/
│   └── MazeContext.js          # Context API để quản lý trạng thái
│
├── hooks/
│   ├── useMazeGenerator.js     # Custom hook cho quá trình sinh mê cung
│   └── useMazeSolver.js        # Custom hook cho quá trình giải mê cung
│
├── styles/                     # Các file CSS
│   ├── App.css                 # Styles chung
│   ├── index.css               # Styles cơ bản
│   ├── Header.css              # Styles cho Header
│   ├── Footer.css              # Styles cho Footer
│   ├── Navigation.css          # Styles cho Navigation
│   ├── MazeGenerator.css       # Styles cho trang sinh mê cung
│   └── MazeSolver.css          # Styles cho trang giải mê cung
│
├── App.js                      # Component chính
└── index.js                    # Điểm khởi đầu ứng dụng
```

## Chức năng

### Trang Sinh mê cung
- Sinh mê cung sử dụng 3 thuật toán: DFS, Prim và Wilson
- Tùy chỉnh kích thước mê cung
- Điều khiển tốc độ mô phỏng
- Điều khiển từng bước quá trình sinh mê cung
- Lưu mê cung vào localStorage để sử dụng tại trang Giải mê cung

### Trang Giải mê cung
- Sử dụng các thuật toán học tăng cường để giải mê cung
- Chọn giữa Q-Learning và SARSA
- Tải mê cung đã lưu hoặc tạo mê cung mới
- Hiển thị quá trình chuột di chuyển tìm phô mai
- Hiển thị heatmap giá trị Q
- Hiển thị thống kê về hiệu suất giải mê cung

## Yêu cầu backend

Để sử dụng đầy đủ chức năng của ứng dụng, cần chạy server backend tại http://localhost:8000.

## Cấu hình API

API endpoint có thể được cấu hình trong file `src/api/mazeApi.js`.