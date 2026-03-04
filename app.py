import streamlit as st
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import tempfile
from scipy import signal as scipy_signal
import pandas as pd
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="工业机器人声纹监测系统",
    page_icon="🤖",
    layout="wide"
)

# ==================== 标题 ====================
st.title("🤖 工业机器人臂声纹智能监测系统")
st.markdown("---")

# ==================== 模型定义（和训练一致） ====================
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim=40):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==================== 工具函数 ====================
@st.cache_resource
def load_model_and_threshold():
    """加载模型和阈值（只加载一次，缓存起来）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Autoencoder(input_dim=40)
    model.load_state_dict(torch.load('autoencoder_model.pth', map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    
    threshold = np.load('final_threshold.npy')
    
    return model, threshold, device

def bandpass_filter(signal, low_freq, high_freq, sr, order=5):
    """带通滤波器"""
    nyquist = 0.5 * sr
    low = max(low_freq / nyquist, 0.001)
    high = min(high_freq / nyquist, 0.999)
    
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal)
    
    return filtered

def compute_frame_error(frame, model, device):
    """计算单帧误差"""
    mfccs = librosa.feature.mfcc(y=frame, sr=16000, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    feature_tensor = torch.FloatTensor(mfccs_mean).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed = model(feature_tensor)
        mse = torch.mean((reconstructed - feature_tensor) ** 2).item()
    
    return mse

def analyze_audio(audio_path, model, threshold, device):
    """分析音频文件，返回结果和可视化图表"""
    
    # 加载音频
    signal, sr = librosa.load(audio_path, sr=16000)
    
    # 计算整体误差
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    feature_tensor = torch.FloatTensor(mfccs_mean).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed = model(feature_tensor)
        overall_error = torch.mean((reconstructed - feature_tensor) ** 2).item()
    
    # 判断
    is_anomaly = overall_error > threshold
    
    # ========== 修正后的置信度计算公式 ==========
    if is_anomaly:
        # 异常样本：超过阈值越多越确定
        confidence = min(100 * ((overall_error - threshold) / threshold), 99)
    else:
        # 正常样本：离阈值越远越确定
        confidence = 100 * (1 - (overall_error / threshold))
    
    # 计算逐帧误差（用于可视化）
    window_size = 2048
    hop_length = 512
    errors_per_frame = []
    times = []
    
    for i in range(0, len(signal)-window_size, hop_length):
        frame = signal[i:i+window_size]
        error = compute_frame_error(frame, model, device)
        errors_per_frame.append(error)
        times.append(i / sr)
    
    # 找出异常区域
    anomalous_frames = [i for i, e in enumerate(errors_per_frame) if e > threshold]
    time_segments = []
    
    if anomalous_frames:
        segments = []
        current_segment = [anomalous_frames[0]]
        
        for i in range(1, len(anomalous_frames)):
            if anomalous_frames[i] == anomalous_frames[i-1] + 1:
                current_segment.append(anomalous_frames[i])
            else:
                segments.append(current_segment)
                current_segment = [anomalous_frames[i]]
        segments.append(current_segment)
        
        for seg in segments:
            start_time = seg[0] * hop_length / sr
            end_time = (seg[-1] + 1) * hop_length / sr
            time_segments.append((start_time, end_time))
    
    # 创建可视化图表
    fig = plt.figure(figsize=(12, 8))
    
    # 子图1：波形图
    plt.subplot(3, 1, 1)
    time_axis = np.linspace(0, len(signal)/sr, len(signal))
    plt.plot(time_axis, signal, color='blue', alpha=0.7, linewidth=0.5)
    
    for start, end in time_segments:
        plt.axvspan(start, end, color='red', alpha=0.3)
    
    plt.xlabel('时间 (秒)')
    plt.ylabel('振幅')
    plt.title(f'波形图 - 整体误差: {overall_error:.4f}')
    plt.grid(True, alpha=0.3)
    
    # 子图2：频谱图
    plt.subplot(3, 1, 2)
    plt.specgram(signal, Fs=sr, NFFT=1024, noverlap=512, cmap='viridis')
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')
    plt.title('频谱图')
    plt.colorbar(label='强度 (dB)')
    
    for start, end in time_segments:
        plt.axvspan(start, end, color='red', alpha=0.3)
    
    # 子图3：逐帧误差
    plt.subplot(3, 1, 3)
    plt.plot(times, errors_per_frame, color='blue', linewidth=1)
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'阈值 ({threshold:.2f})')
    plt.xlabel('时间 (秒)')
    plt.ylabel('重构误差')
    plt.title('逐帧误差曲线')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    for start, end in time_segments:
        plt.axvspan(start, end, color='red', alpha=0.3)
    
    plt.tight_layout()
    
    return {
        'is_anomaly': is_anomaly,
        'overall_error': overall_error,
        'threshold': threshold,
        'confidence': confidence,
        'time_segments': time_segments,
        'figure': fig
    }

# ==================== 加载模型 ====================
with st.spinner("加载模型中..."):
    try:
        model, threshold, device = load_model_and_threshold()
        st.sidebar.success("✅ 模型加载成功")
        st.sidebar.info(f"设备: {device}")
        st.sidebar.info(f"阈值: {threshold:.4f}")
    except Exception as e:
        st.sidebar.error(f"❌ 模型加载失败: {e}")
        st.stop()

# ==================== 侧边栏信息 ====================
with st.sidebar:
    st.header("📌 项目信息")
    st.markdown("""
    **工业机器人声纹监测系统**
    
    - 基于自编码器的无监督异常检测
    - 训练数据：DCASE 2024 Challenge
    - 检测设备：工业机器人臂
    - 最佳阈值：97.56分位数
    - 测试集异常率：32%
    """)
    
    st.markdown("---")
    st.header("📊 置信度说明")
    st.markdown("""
    **置信度含义：**
    
    置信度表示模型判断的确定程度（0-100%），**越高越确定**。
    
    - **正常样本**（误差 < 阈值）
      - >80%：非常确定是正常
      - 50-80%：较确定是正常
      - <50%：不确定，接近异常边界
    
    - **异常样本**（误差 > 阈值）
      - >50%：非常确定是异常
      - 20-50%：较确定是异常
      - <20%：不确定，刚过异常边界
    
    **注意**：置信度只反映确定程度，不反映严重程度。严重程度要看误差超过阈值多少。
    """)

# ==================== 主界面：单文件测试 ====================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传音频文件")
    
    uploaded_file = st.file_uploader(
        "选择WAV格式的音频文件",
        type=['wav'],
        help="上传工业机器人臂的运行声音（支持单声道16kHz WAV格式）"
    )
    
    # 示例文件选择
    st.markdown("---")
    st.subheader("🎯 或选择示例文件")
    
    example_files = [
        "section_00_0000.wav (正常样本)",
        "section_00_0006.wav (异常样本)"
    ]
    selected_example = st.selectbox("选择示例", ["无"] + example_files)
    
    # 确定要分析的音频
    audio_to_analyze = None
    
    if uploaded_file is not None:
        # 保存上传的文件到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_to_analyze = tmp_file.name
        st.audio(uploaded_file)
        st.success(f"已上传: {uploaded_file.name}")
    
    elif selected_example != "无":
        # 使用示例文件
        example_name = selected_example.split(" ")[0]
        example_path = f"data/test/{example_name}"
        if os.path.exists(example_path):
            audio_to_analyze = example_path
            st.audio(example_path)
            st.info(f"已选择示例: {selected_example}")
        else:
            st.error(f"示例文件不存在: {example_path}")

with col2:
    st.subheader("🔍 检测结果")
    
    if audio_to_analyze and st.button("开始分析", type="primary"):
        with st.spinner("分析中，请稍候..."):
            try:
                # 分析音频
                result = analyze_audio(audio_to_analyze, model, threshold, device)
                
                # ========== 显示结果卡片 ==========
                if result['is_anomaly']:
                    st.error(f"❌ 检测结果：异常")
                else:
                    st.success(f"✅ 检测结果：正常")
                
                # 指标卡片
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("重构误差", f"{result['overall_error']:.4f}")
                with col_b:
                    st.metric("阈值", f"{result['threshold']:.4f}")
                with col_c:
                    st.metric("置信度", f"{result['confidence']:.1f}%")
                
                # ========== 置信度解读 ==========
                st.markdown("---")
                st.markdown("**📊 置信度解读**")
                
                # 进度条
                col_p1, col_p2 = st.columns([3, 1])
                with col_p1:
                    if result['is_anomaly']:
                        bar_color = "#ff4b4b"  # 红色
                    else:
                        bar_color = "#00cc66"  # 绿色
                    
                    st.markdown(f"""
                    <div style="width: 100%; background-color: #f0f0f0; border-radius: 20px; margin: 10px 0;">
                        <div style="width: {result['confidence']}%; background-color: {bar_color}; 
                                  border-radius: 20px; padding: 8px 0; text-align: center; color: white; font-weight: bold;">
                            {result['confidence']:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_p2:
                    if result['is_anomaly']:
                        if result['confidence'] > 50:
                            st.markdown("**🔴 非常确定是异常**")
                        elif result['confidence'] > 20:
                            st.markdown("**🟠 较确定是异常**")
                        else:
                            st.markdown("**🟡 不确定（刚过阈值）**")
                    else:
                        if result['confidence'] > 80:
                            st.markdown("**🟢 非常确定是正常**")
                        elif result['confidence'] > 50:
                            st.markdown("**🔵 较确定是正常**")
                        else:
                            st.markdown("**🟡 不确定（接近阈值）**")
                
                # 置信度详细说明
                with st.expander("📋 置信度详细说明"):
                    st.markdown("""
                    **置信度含义：**
                    
                    置信度表示模型判断的确定程度（0-100%），**越高越确定**。
                    
                    - **正常样本**（误差 < 阈值）
                      - >80%：非常确定是正常
                      - 50-80%：较确定是正常
                      - <50%：不确定，接近异常边界
                    
                    - **异常样本**（误差 > 阈值）
                      - >50%：非常确定是异常
                      - 20-50%：较确定是异常
                      - <20%：不确定，刚过异常边界
                    
                    **注意**：置信度只反映确定程度，不反映严重程度。严重程度要看误差超过阈值多少。
                    """)
                
                # 异常区域信息
                if result['time_segments']:
                    st.info(f"发现 {len(result['time_segments'])} 个异常时间段")
                    
                    # 显示时间段表格
                    segments_data = []
                    for i, (start, end) in enumerate(result['time_segments'][:5]):  # 只显示前5个
                        segments_data.append({
                            "段号": i+1,
                            "开始(秒)": f"{start:.2f}",
                            "结束(秒)": f"{end:.2f}",
                            "持续(秒)": f"{end-start:.2f}"
                        })
                    st.dataframe(pd.DataFrame(segments_data))
                
                # 显示图表
                st.pyplot(result['figure'])
                
                # 清除临时文件
                if uploaded_file is not None and os.path.exists(audio_to_analyze):
                    os.unlink(audio_to_analyze)
                    
            except Exception as e:
                st.error(f"分析出错: {e}")
    else:
        st.info("请在左侧上传音频或选择示例文件")

# ==================== 批量测试功能 ====================
st.markdown("---")
st.subheader("📦 批量测试")

with st.expander("点击展开批量测试功能", expanded=False):
    st.markdown("""
    **批量上传多个音频文件**，系统将自动分析并生成汇总报告。
    支持同时上传最多20个WAV文件。
    """)
    
    # 批量文件上传
    uploaded_files = st.file_uploader(
        "选择多个WAV格式的音频文件",
        type=['wav'],
        accept_multiple_files=True,
        help="可同时选择多个文件（按住Ctrl多选）"
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        st.info(f"已选择 {len(uploaded_files)} 个文件")
        
        # 显示文件列表
        file_list = []
        for f in uploaded_files:
            file_list.append({"文件名": f.name, "大小": f"{f.size/1024:.1f} KB"})
        st.dataframe(pd.DataFrame(file_list))
        
        # 开始批量测试按钮
        if st.button("🚀 开始批量测试", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 创建临时目录存放上传的文件
            temp_dir = tempfile.mkdtemp()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"正在处理: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                
                # 保存临时文件
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    # 分析音频
                    result = analyze_audio(temp_path, model, threshold, device)
                    
                    # 根据置信度添加确定程度标签
                    if result['is_anomaly']:
                        if result['confidence'] > 50:
                            certainty = "非常确定"
                        elif result['confidence'] > 20:
                            certainty = "较确定"
                        else:
                            certainty = "不确定"
                    else:
                        if result['confidence'] > 80:
                            certainty = "非常确定"
                        elif result['confidence'] > 50:
                            certainty = "较确定"
                        else:
                            certainty = "不确定"
                    
                    # 保存结果
                    results.append({
                        "文件名": uploaded_file.name,
                        "判断结果": "异常" if result['is_anomaly'] else "正常",
                        "确定程度": certainty,
                        "重构误差": f"{result['overall_error']:.4f}",
                        "阈值": f"{result['threshold']:.4f}",
                        "置信度": f"{result['confidence']:.1f}%",
                        "异常段数": len(result['time_segments'])
                    })
                    
                except Exception as e:
                    results.append({
                        "文件名": uploaded_file.name,
                        "判断结果": "错误",
                        "确定程度": "-",
                        "重构误差": "-",
                        "阈值": "-",
                        "置信度": "-",
                        "异常段数": "-"
                    })
                
                # 更新进度条
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # 清理临时文件
                os.unlink(temp_path)
            
            os.rmdir(temp_dir)
            status_text.text("✅ 批量测试完成！")
            
            # 显示汇总结果
            st.markdown("---")
            st.subheader("📊 批量测试结果汇总")
            
            # 统计
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # 统计信息
            total = len(results)
            normal_count = len([r for r in results if r['判断结果'] == '正常'])
            anomaly_count = len([r for r in results if r['判断结果'] == '异常'])
            error_count = len([r for r in results if r['判断结果'] == '错误'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总文件数", total)
            with col2:
                st.metric("正常", normal_count)
            with col3:
                st.metric("异常", anomaly_count)
            with col4:
                st.metric("异常率", f"{anomaly_count/total*100:.1f}%" if total > 0 else "0%")
            
            # 置信度分布统计
            with st.expander("📊 置信度分布"):
                confidence_levels = []
                for r in results:
                    if r['判断结果'] == '正常' and r['判断结果'] != '错误':
                        conf = float(r['置信度'].rstrip('%'))
                        if conf > 80:
                            confidence_levels.append("非常确定 (>80%)")
                        elif conf > 50:
                            confidence_levels.append("较确定 (50-80%)")
                        else:
                            confidence_levels.append("不确定 (<50%)")
                    elif r['判断结果'] == '异常':
                        conf = float(r['置信度'].rstrip('%'))
                        if conf > 50:
                            confidence_levels.append("非常确定 (>50%)")
                        elif conf > 20:
                            confidence_levels.append("较确定 (20-50%)")
                        else:
                            confidence_levels.append("不确定 (<20%)")
                
                if confidence_levels:
                    conf_df = pd.DataFrame({"置信度等级": confidence_levels})
                    conf_counts = conf_df["置信度等级"].value_counts().reset_index()
                    conf_counts.columns = ["置信度等级", "数量"]
                    st.dataframe(conf_counts)
            
            # 导出结果
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载测试结果 (CSV)",
                data=csv,
                file_name=f"batch_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# ==================== 底部说明 ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>基于DCASE 2024 Challenge数据集 | 挑战杯参赛项目 | 阈值: 97.56分位数 | 异常率: 32%</p>
</div>
""", unsafe_allow_html=True)