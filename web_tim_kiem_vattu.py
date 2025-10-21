import streamlit as st
import pandas as pd
import requests
from io import StringIO
import numpy as np
import base64
import os
import sys 
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# === CONFIG GOOGLE SHEET ===
GSHEET_URL = "https://docs.google.com/spreadsheets/d/19Jcdw6ONUgRRj1IYcD7m8XruMAdrMReg/export?format=csv&gid=759908915"

# === CÁC ĐƯỜNG DẪN ẢNH CỤC BỘ (HÃY ĐẢM BẢO PHẦN MỞ RỘNG ĐÚNG) ===
HINHNEN_PATH = "images/HINHNEN.png"   
LOGO_PATH = "images/Logo-VNPT.png"     


DISPLAY_COLUMNS = [
    'Ngày nhận', 'KHO', 'TÊN VẬT TƯ', 'Thiết bị', 'ĐVT', 'SL', 'MÃ VT',
    'Serial', 'Serial-ĐHSX', 'CĂN CỨ (Công văn)', 'DP/Giao', 'GHI CHÚ', 'Người cập nhật'
]
SEARCH_COLUMNS = ['TÊN VẬT TƯ', 'Serial']


# === HÀM TẢI ẢNH TỪ CỤC BỘ VÀ CHUYỂN SANG BASE64 ===
def get_base64_image(image_path):
    """Tải ảnh từ cục bộ bằng đường dẫn tương đối."""
    current_path = ""
    
    if os.path.exists(image_path):
        current_path = image_path
    else:
        # Xử lý đường dẫn tuyệt đối (dự phòng)
        base_path = os.path.dirname(os.path.abspath(__file__)) if os.path.exists(os.path.abspath(__file__)) else os.getcwd()
        full_path = os.path.join(base_path, image_path)
        if os.path.exists(full_path):
             current_path = full_path

    if not current_path:
        return ""

    try:
        with open(current_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""


# === CLEAN COLUMN ===
def clean_and_normalize_column(col_name):
    if pd.isna(col_name):
        return ""
    cleaned_name = str(col_name).strip().replace('\ufeff', '').replace('\xa0', '')
    return cleaned_name.strip()

# === CSV CONVERT ===
def convert_df_to_csv(df):
    df_temp = df.copy()
    for col in df_temp.columns:
        if df_temp[col].dtype == np.int64:
            df_temp[col] = df_temp[col].astype(int) 
    return df_temp.to_csv(index=False).encode('utf-8')

# === LOAD DATA (DÙNG CACHE MỚI) ===
# Dùng @st.cache_data (yêu cầu Streamlit >= 1.18.0)
@st.cache_data(ttl=300, show_spinner="Đang tải dữ liệu từ Google Sheet...") 
def load_data(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    df = pd.read_csv(StringIO(response.text), encoding='utf-8', header=0)
    df.columns = df.columns.astype(str).map(clean_and_normalize_column)
    df.dropna(how='all', inplace=True)

    if 'SL' in df.columns:
        df['SL'] = pd.to_numeric(df['SL'], errors='coerce').fillna(0).astype(np.int64)
    if 'Ngày nhận' in df.columns:
        df['Ngày nhận DT'] = pd.to_datetime(df['Ngày nhận'], errors='coerce', dayfirst=True)
        df['Ngày nhận'] = df['Ngày nhận DT'].dt.strftime('%d/%m/%Y').fillna('Không rõ')
    return df

# === RESET FILTER ===
def reset_filters():
    for key in ['vattu_filter', 'date_filter', 'search_query', 'input_search', 'vattu_filter_old', 'date_filter_old']:
        if 'filter' in key or 'old' in key:
            st.session_state[key] = 'Tất cả'
        elif 'search' in key:
            st.session_state[key] = ""

# === HÀM XÓA CACHE ĐỘC LẬP VÀ RELOAD (DÙNG CÚ PHÁP MỚI) ===
def clear_cache_and_rerun():
    # Sử dụng phương thức xóa cache mới và an toàn nhất
    st.cache_data.clear() 
    st.cache_resource.clear()
    st.rerun() 
    
def app():
    st.set_page_config(page_title="Tìm Kiếm Vật Tư", layout="wide", initial_sidebar_state="expanded") 
    st_autorefresh(interval=300000, key="data_refresh") # Tự động làm mới 5 phút

    # === TẢI HÌNH NỀN VÀ LOGO BẰNG BASE64 ===
    HINHNEN_BASE64 = get_base64_image(HINHNEN_PATH)
    LOGO_BASE64 = get_base64_image(LOGO_PATH)
    
    # === XÁC ĐỊNH ĐỊNH DẠNG MIME TYPE CHO HÌNH NỀN ===
    mime_type = "image/png" 
    if HINHNEN_PATH.lower().endswith(('.jpg', '.jpeg')):
        mime_type = "image/jpeg"

    # === KHỐI CSS CHUNG ===
    
    # 1. CSS Cho Background
    if HINHNEN_BASE64:
        background_style = f"""
            <style>
            body {{
                background-image: url("data:{mime_type};base64,{HINHNEN_BASE64}"); 
                background-size: cover; 
                background-attachment: fixed; 
                background-position: center;
            }}
            /* Đảm bảo các thành phần Streamlit trong suốt */
            .stApp, .main, [data-testid="stSidebar"] > div, footer {{ 
                background-color: transparent !important; 
            }}
            </style>
            """
        st.markdown(background_style, unsafe_allow_html=True)

    # 2. CSS Chung Cho Tiêu đề, Hộp, etc.
    general_styles = f"""
        <style>
        
        /* === CSS CHO H1 (Gộp hai dòng chính và logo) === */
        h1 {{
            text-align: center;
            color: #004780 !important; 
            font-weight: 900 !important; 
            font-size: 3rem; 
            margin: 0 !important;
            line-height: 1.1; 
            padding-top: 20px; 
        }}
        h1 .subtitle {{
            display: block; 
            font-weight: bold !important;
            font-size: 1.8rem; 
            color: #004780 !important;
            margin-top: 0.1em; 
            margin-bottom: 0 !important;
        }}
        /* === CSS CHO H2 (Tiêu đề Tab và Thống kê) === */
        h2 {{ 
            color: #004780 !important; 
            font-weight: bold;
            font-size: 2rem; 
            text-align: left !important;
            margin: 0.5em 0 0.2em !important; 
        }}
        /* === CSS CHO H3 (Tìm kiếm) === */
        h3 {{ 
            color: #004780 !important; 
            font-weight: bold;
            font-size: 1.8rem;
            margin: 0.1em 0 0.2em !important;
        }}
        /* === CSS CHO H4 (Dữ liệu sau lọc) === */
        h4 {{
            color: #004780 !important;
            font-weight: bold;
            font-size: 1.6rem;
            text-align: left !important;
            margin: 0.5em 0 0.2em !important;
        }}
        /* === SỬA KHOẢNG CÁCH CỦA DẤU GẠCH NGANG (HR) === */
        hr {{
            margin-top: 0.5rem !important;
            margin-bottom: 1rem !important;
        }}
        </style>
    """
    st.markdown(general_styles, unsafe_allow_html=True)

    # === HEADER (H1, Logo và H1 subtitle) ===
    col_logo, col_title = st.columns([1, 8])

    with col_logo:
        # Logo VNPT (dùng Base64 cho độ tin cậy cao nhất)
        if LOGO_BASE64:
             st.markdown(
                 f"""
                 <img src="data:image/png;base64,{LOGO_BASE64}" width="100">
                 """, unsafe_allow_html=True
             )
        else:
             # Biện pháp dự phòng nếu không tìm thấy file logo
             st.markdown("<h2 style='text-align:center;'>🌐</h2>", unsafe_allow_html=True)
            
    with col_title:
        # H1: Gộp cả hai dòng
        st.markdown(
            f"""
            <h1>
                Ứng Dụng Quản Lý & Tra Cứu Vật Tư Vô Tuyến 
                <span class="subtitle">Trạm Viễn Thông Đắk Nông</span>
            </h1>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # === INIT SESSION ===
    defaults = {'search_query':"", 'vattu_filter':'Tất cả', 'date_filter':'Tất cả',
                'vattu_filter_old':'Tất cả', 'date_filter_old':'Tất cả', 'input_search':''}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    # === LOAD DATA ===
    try:
        df = load_data(GSHEET_URL)
    except Exception as e:
        st.error(f"❌ Lỗi tải dữ liệu. Vui lòng kiểm tra kết nối mạng và URL: {e}")
        return
        
    if df.empty:
        st.error("❌ Không tải được dữ liệu từ Google Sheet.")
        return
        
    df_filtered = df.copy()

    # === SIDEBAR FILTER ===
    st.sidebar.title("🛠️ Bộ Lọc")

    if 'TÊN VẬT TƯ' in df.columns:
        list_vattu = ['Tất cả'] + sorted(df['TÊN VẬT TƯ'].dropna().unique())
        st.sidebar.selectbox("Tên Vật Tư:", list_vattu, key='vattu_filter') 
        
        if st.session_state.vattu_filter != st.session_state.vattu_filter_old:
            st.session_state.date_filter = 'Tất cả'
            st.session_state.vattu_filter_old = st.session_state.vattu_filter
            st.rerun() 

        if st.session_state.vattu_filter != 'Tất cả':
            df_filtered = df_filtered[df_filtered['TÊN VẬT TƯ'] == st.session_state.vattu_filter]

    if 'Ngày nhận' in df.columns:
        date_list = sorted(df_filtered['Ngày nhận'].loc[df_filtered['Ngày nhận'] != 'Không rõ'].unique(),
                             key=lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'),
                             reverse=True)
        date_list = ['Tất cả'] + date_list
        st.sidebar.selectbox("Ngày nhận:", date_list, key='date_filter')
        if st.session_state.date_filter != 'Tất cả':
            df_filtered = df_filtered[df_filtered['Ngày nhận'] == st.session_state.date_filter]

    st.sidebar.markdown("---")
    
    col_ref, col_reset = st.sidebar.columns(2)
    
    # Nút F5 sẽ gọi hàm xóa cache an toàn
    if col_ref.button("🔄 F5"): 
        clear_cache_and_rerun()
        
    if col_reset.button("🗑️ Xóa"): 
        reset_filters()
        st.rerun() 

    st.sidebar.metric("Tổng vật tư (Gốc)", f"{df['SL'].sum():,} cái")
    st.sidebar.caption(f"✅ {len(df)} dòng | Tự động cập nhật mỗi 5 phút")


    # === H2: TIÊU ĐỀ KHỐI TAB VÀ PHÂN CHIA TAB ===
    st.markdown("<h2>📊 KHỐI TRA CỨU & THỐNG KÊ</h2>", unsafe_allow_html=True) 

    tab1, tab2 = st.tabs(["🔍 TRA CỨU & LỌC", "📊 THỐNG KÊ NHANH"])

    # ====================================================
    #           TAB 1: TRA CỨU & LỌC
    # ====================================================
    with tab1: 
        # H3: Tiêu đề Tìm kiếm
        st.subheader("🔍 Tìm Kiếm Vật Tư hoặc Serial")
        
        # Hộp tìm kiếm (Sử dụng 1 cột duy nhất, không có nút Xóa)
        tu_khoa = st.text_input("Nhập từ khóa:", placeholder="Ví dụ: AHEB, L615...",
                                        value=st.session_state.search_query).strip()
        
        if tu_khoa != st.session_state.search_query:
            st.session_state.search_query = tu_khoa
            st.rerun()
        
        cols_to_display = [c for c in DISPLAY_COLUMNS if c in df_filtered.columns]

        if st.session_state.search_query:
            tu_khoa = st.session_state.search_query
            mask = pd.Series(False, index=df_filtered.index)
            for c in [x for x in SEARCH_COLUMNS if x in df_filtered.columns]:
                if c in df_filtered.columns:
                    mask |= df_filtered[c].astype(str).str.contains(tu_khoa, case=False, na=False)
            result = df_filtered[mask]
            if not result.empty:
                # H4: Tiêu đề Kết quả
                st.markdown(f"<h4>Kết Quả: '{tu_khoa}' ({len(result)} dòng)</h4>", unsafe_allow_html=True)
                st.dataframe(result[cols_to_display], height=500) 
                st.download_button("📥 Tải Kết Quả", convert_df_to_csv(result),
                                    file_name='KetQua_Tim_Kiem.csv', mime='text/csv')
            else:
                st.warning("Không tìm thấy kết quả nào.")
        else:
            # H4: Tiêu đề Dữ liệu sau khi lọc
            st.markdown("<h4>📋 Dữ Liệu Sau Khi Lọc</h4>", unsafe_allow_html=True)
            
            st.dataframe(df_filtered[cols_to_display], height=500) 
            st.download_button("📥 Tải Dữ Liệu", convert_df_to_csv(df_filtered),
                                file_name='DuLieu_Loc.csv', mime='text/csv')

    # ====================================================
    #           TAB 2: THỐNG KÊ NHANH
    # ====================================================
    with tab2: 
        # H2: Tiêu đề Thống kê (sử dụng st.header, đã căn trái trong CSS)
        st.header("📈 Thống Kê Tổng Quan Vật Tư")

        if 'SL' in df.columns:
            total_sl = df['SL'].sum()
            total_rows = len(df)
            col_t1, col_t2 = st.columns(2)
            col_t1.metric("Tổng Số Lượng Vật Tư (Gốc)", f"{total_sl:,} cái")
            col_t2.metric("Tổng Số Dòng Dữ Liệu", f"{total_rows:,} dòng")
        
        st.markdown("---")

        if 'TÊN VẬT TƯ' in df.columns and 'SL' in df.columns:
            tk_vattu = df.groupby('TÊN VẬT TƯ')['SL'].sum().reset_index()
            tk_vattu.rename(columns={'SL': 'Tổng Số Lượng'}, inplace=True) 

            # H3: Tiêu đề Phân phối
            st.subheader("Phân Phối Số Lượng Theo Loại Vật Tư (Top 10)")
            fig_pie = px.pie(tk_vattu.head(10), values='Tổng Số Lượng', names='TÊN VẬT TƯ',
                              title='Top 10 Loại Vật Tư Theo Số Lượng', hole=.3)
            st.plotly_chart(fig_pie)

            st.download_button("📥 Tải Bảng Thống Kê Loại", data=convert_df_to_csv(tk_vattu),
                                file_name='ThongKe_Loai_VatTu.csv', mime='text/csv')

        st.markdown("---")

        if 'Ngày nhận DT' in df.columns: 
            df_temp = df.copy()
            
            seven_days_ago = pd.Timestamp('today').normalize() - pd.Timedelta(days=7)
            
            recent_updates = df_temp[df_temp['Ngày nhận DT'] >= seven_days_ago]
            updates_daily = recent_updates.groupby('Ngày nhận DT').size().reset_index(name='Số Lần Cập Nhật')
            
            # H3: Tiêu đề Cập nhật
            st.subheader("Số Lần Cập Nhật Dữ Liệu Trong 7 Ngày Gần Nhất")
            if not updates_daily.empty:
                updates_daily['Ngày nhận DT'] = updates_daily['Ngày nhận DT'].dt.strftime('%d/%m')
                fig_line = px.line(updates_daily, x='Ngày nhận DT', y='Số Lần Cập Nhật', 
                                    title='Tần suất Cập Nhật (7 Ngày)', markers=True)
                st.plotly_chart(fig_line)
            else:
                st.info("Chưa có cập nhật nào trong 7 ngày gần nhất.")


if __name__ == "__main__":
    app()