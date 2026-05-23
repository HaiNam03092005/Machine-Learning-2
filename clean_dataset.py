import pandas as pd
import glob
import os

# 1. Đường dẫn tới thư mục chứa các file .xls của bạn
folder_path = 'nba_history/' 

# 2. Lấy danh sách tất cả các file có đuôi .xls hoặc .xlsx trong thư mục đó
# Dấu * đại diện cho tên file bất kỳ
file_list = glob.glob(os.path.join(folder_path, "*.xls*"))

# Tạo một danh sách rỗng để chứa dữ liệu của từng file
all_seasons_list = []

# 3. Vòng lặp đọc từng file và thêm cột năm (Season)
for file_path in file_list:
    # Lấy tên file gốc (ví dụ: '2011.xls')
    file_name = os.path.basename(file_path)
    
    # Trích xuất năm từ tên file (Cắt bỏ đuôi '.xls' và chuyển thành số nguyên)
    # Ví dụ: '2011.xls' tách ra thành '2011' -> 2011
    season_year = int(file_name.split('.')[0]) 
    
    # Đọc file excel bằng Pandas
    df_season = pd.read_excel(file_path)
    
    # QUAN TRỌNG: Tạo thêm cột 'Season' cho file này
    # Để khi gộp lại, máy phân biệt được hàng nào thuộc mùa giải nào
    df_season['Season'] = season_year
    
    # Thêm DataFrame của mùa giải này vào danh sách tổng
    all_seasons_list.append(df_season)

# 4. GỘP TẤT CẢ THÀNH 1 FILE DUY NHẤT (Master DataFrame)
master_df = pd.concat(all_seasons_list, ignore_index=True)

print(f"🔥 Đã gộp thành công {len(file_list)} file mùa giải vào Pandas!")
print(f"Tổng số hàng dữ liệu: {master_df.shape[0]}")