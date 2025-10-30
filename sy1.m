%% 实验一
clear all; close all; clc;

% % 3.1 录制语音
% Fs = 16000;
% fprintf('=== 3.1 语音录制 ===\n');
% 录制语音
% recObj = audiorecorder(16000, 16, 1);
% disp('开始录音，请说"长江大学"...');
% recordblocking(recObj, 2);
% disp('录音结束');
% myRecording = getaudiodata(recObj);
% audiowrite('yeshun222.wav', myRecording, Fs);
% fprintf('语音已保存为 yeshun222.wav\n');

% 使用现有语音文件
[speech, Fs] = audioread('yeshun.wav');%speech为原始语音数据序列
fprintf('采样率: %d Hz\n', Fs);
fprintf('语音长度: %.2f 秒\n', length(speech)/Fs);%时长=采样点数/采样率

% 截取有效部分
x = speech(10000:30000);
nx = length(x);

% 绘制原始语音
% figure('Position', [100, 100, 1200, 400]);
figure('Position', [1, 1, 1920, 1080], 'Name', '原始语音信号', 'NumberTitle', 'on');
plot((0:length(x)-1)/Fs, x);
title('3.1 原始语音信号 "长江大学"');
xlabel('时间 (s)'); ylabel('幅度');
grid on;

%% 3.2 分帧处理
fprintf('\n=== 3.2 分帧处理 ===\n');
nwin = round(0.03 * Fs);  % 帧长30ms
ninc = round(0.01 * Fs);  % 帧移10ms
nf = fix((nx - nwin + ninc) / ninc);
fprintf('帧长: %d 采样点 (%.1f ms)\n', nwin, nwin/Fs*1000);
fprintf('帧移: %d 采样点 (%.1f ms)\n', ninc, ninc/Fs*1000);
fprintf('总帧数: %d\n', nf);

% 分帧
frame = zeros(nf, nwin);
for i = 1:nf
    start_index = (i-1) * ninc + 1;
    end_index = start_index + nwin - 1;
    if end_index <= nx
        frame(i, :) = x(start_index:end_index);
    end
end

% 加汉明窗
w = hamming(nwin)';
frame = frame .* w;

fprintf('分帧完成，使用汉明窗\n');

%% 3.3 短时能量和过零率分析
fprintf('\n=== 3.3 短时能量和过零率分析 ===\n');

% 计算短时能量
shortenergy = sum(frame.^2, 2);

% 计算短时过零率
nzcr = zeros(nf, 1);
for i = 1:nf
    current_frame = frame(i, :);
    for j = 1:nwin-1
        if current_frame(j) * current_frame(j+1) < 0
            nzcr(i) = nzcr(i) + 1;
        end
    end
end

% 设置阈值
energy_threshold = 0.1 * max(shortenergy);%短时能量阈值用于区分有声段和静音段
zcr_threshold = 0.3 * max(nzcr);%

% 浊音/清音判断
voice_decision = zeros(nf, 1);  % 1:浊音, 0:清音, -1:静音
for i = 1:nf
    if shortenergy(i) > energy_threshold
        if nzcr(i) < zcr_threshold
            voice_decision(i) = 1;  % 浊音
        else
            voice_decision(i) = 0;  % 清音
        end
    else
        voice_decision(i) = -1;     % 静音
    end
end

% 可视化短时分析结果
figure('Position', [1, 1, 1920, 1080], 'Name', '短时能量分析', 'NumberTitle', 'on');

% 原始语音
subplot(4,1,1);
t_original = (0:length(x)-1) / Fs;
plot(t_original, x);
title('3.3.1 原始语音信号');
xlabel('时间 (s)'); ylabel('幅度');
grid on;

% 短时能量
subplot(4,1,2);
t_frames = (1:nf) * ninc / Fs;
plot(t_frames, shortenergy, 'b-', 'LineWidth', 1.5);
hold on;
plot([t_frames(1), t_frames(end)], [energy_threshold, energy_threshold], 'r--', 'LineWidth', 1);
title('3.3.2 短时能量分析');
xlabel('时间 (s)'); ylabel('能量');
legend('短时能量', '能量阈值', 'Location', 'northeast');
grid on;

% 短时过零率：信号每秒钟穿过零轴的次数
%浊音：过零率低（周期性信号，缓慢变化）
%清音：过零率高（类噪声信号，快速振荡）
subplot(4,1,3);
plot(t_frames, nzcr, 'g-', 'LineWidth', 1.5);
hold on;
plot([t_frames(1), t_frames(end)], [zcr_threshold, zcr_threshold], 'r--', 'LineWidth', 1);
title('3.3.3 短时过零率分析');
xlabel('时间 (s)'); ylabel('过零率');
legend('短时过零率', '过零率阈值', 'Location', 'northeast');
grid on;

subplot(4,1,4);
% 绘制浊音/清音区域
t_frame = (1:120) * ninc / Fs;
voiced_frames = t_frame(voice_decision(1:120) == 1);
unvoiced_frames = t_frame(voice_decision(1:120) == 0);
silence_frames = t_frame(voice_decision(1:120) == -1);

hold on;

% 绘制浊音区域（红色）
if ~isempty(voiced_frames)
    for i = 1:length(voiced_frames)
        rectangle('Position', [voiced_frames(i)-ninc/Fs, 0.6, ninc/Fs, 0.3], ...
                 'FaceColor', 'r', 'EdgeColor', 'none');
    end
end

% 绘制清音区域（绿色）
if ~isempty(unvoiced_frames)
    for i = 1:length(unvoiced_frames)
        rectangle('Position', [unvoiced_frames(i)-ninc/Fs, 0.3, ninc/Fs, 0.3], ...
                 'FaceColor', 'g', 'EdgeColor', 'none');
    end
end

% 绘制静音区域（蓝色）
if ~isempty(silence_frames)
    for i = 1:length(silence_frames)
        rectangle('Position', [silence_frames(i)-ninc/Fs, 0, ninc/Fs, 0.3], ...
                 'FaceColor', 'b', 'EdgeColor', 'none');
    end
end

ylim([0 1]);
title('浊音/清音/静音区域判断');
xlabel('时间(s)'); ylabel('语音类型');

% 创建图例 - 使用简单的线条对象
h1 = plot(NaN, NaN, 'Color', 'r', 'LineWidth', 10);
h2 = plot(NaN, NaN, 'Color', 'g', 'LineWidth', 10);
h3 = plot(NaN, NaN, 'Color', 'b', 'LineWidth', 10);

legend([h1, h2, h3], {'浊音', '清音', '静音'}, 'Location', 'northeast');
grid on;

% 统计结果
fprintf('浊音帧数: %d (%.1f%%)\n', sum(voice_decision==1), sum(voice_decision==1)/nf*100);
fprintf('清音帧数: %d (%.1f%%)\n', sum(voice_decision==0), sum(voice_decision==0)/nf*100);
fprintf('静音帧数: %d (%.1f%%)\n', sum(voice_decision==-1), sum(voice_decision==-1)/nf*100);

%% 3.4 短时自相关分析
fprintf('\n=== 3.4 短时自相关分析 ===\n');

% 选取浊音帧和清音帧
voiced_frames = find(voice_decision == 1);
unvoiced_frames = find(voice_decision == 0);

if isempty(voiced_frames)
    voiced_frame_idx = min(50, nf);%如果没有浊音帧，选择第50帧作为默认值
else
    voiced_frame_idx = voiced_frames(ceil(length(voiced_frames)/2));%浊音帧中选择中间位置的一帧
end

if isempty(unvoiced_frames)
    unvoiced_frame_idx = min(20, nf);
else
    unvoiced_frame_idx = unvoiced_frames(ceil(length(unvoiced_frames)/2));
end

fprintf('选取浊音帧: %d\n', voiced_frame_idx);
fprintf('选取清音帧: %d\n', unvoiced_frame_idx);

% 获取帧数据
voiced_data = frame(voiced_frame_idx, :);
unvoiced_data = frame(unvoiced_frame_idx, :);

% 短时自相关函数


% 计算自相关
max_lag = round(0.02 * Fs);  % 20ms
voiced_autocorr = short_time_autocorr(voiced_data, max_lag);
unvoiced_autocorr = short_time_autocorr(unvoiced_data, max_lag);

% 归一化
voiced_autocorr = voiced_autocorr / max(voiced_autocorr);
unvoiced_autocorr = unvoiced_autocorr / max(unvoiced_autocorr);

% 基音周期检测
search_min = round(0.002 * Fs);  % 2ms，对应最高基频
search_max = round(0.02 * Fs);   % 20ms，对应最低基频
[~, max_pos] = max(voiced_autocorr(search_min:search_max));
pitch_period = max_pos + search_min - 1;
pitch_frequency = Fs / pitch_period;%基音频率 = 采样率 / 基音周期

fprintf('基音周期: %d 采样点 (%.2f ms)\n', pitch_period, pitch_period/Fs*1000);
fprintf('基音频率: %.2f Hz\n', pitch_frequency);

% 可视化自相关分析
figure('Position', [1, 1, 1920, 1080], 'Name', '短时自相关分析', 'NumberTitle', 'on');

% 时域波形
subplot(3,2,1);
plot(voiced_data, 'b-', 'LineWidth', 1.5);
title('3.4.1 浊音帧时域波形');
xlabel('采样点'); ylabel('幅度');
grid on;

subplot(3,2,2);
plot(unvoiced_data, 'r-', 'LineWidth', 1.5);
title('3.4.2 清音帧时域波形');
xlabel('采样点'); ylabel('幅度');
grid on;

% 自相关函数
subplot(3,2,3);
lags = 0:max_lag-1;
time_lags = lags / Fs * 1000;  % 转换为毫秒
plot(time_lags, voiced_autocorr, 'b-', 'LineWidth', 1.5);
hold on;
plot(time_lags(pitch_period), voiced_autocorr(pitch_period), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
title('3.4.3 浊音帧自相关函数');
xlabel('延迟 (ms)'); ylabel('归一化自相关');
legend('自相关', '基音周期', 'Location', 'northeast');
grid on;

subplot(3,2,4);
plot(time_lags, unvoiced_autocorr, 'r-', 'LineWidth', 1.5);
title('3.4.4 清音帧自相关函数');
xlabel('延迟 (ms)'); ylabel('归一化自相关');
grid on;

% 对比
subplot(3,2,5);
plot(time_lags, voiced_autocorr, 'b-', 'LineWidth', 1.5);
hold on;
plot(time_lags, unvoiced_autocorr, 'r-', 'LineWidth', 1.5);
title('3.4.5 浊音 vs 清音自相关对比');
xlabel('延迟 (ms)'); ylabel('归一化自相关');
legend('浊音', '清音', 'Location', 'northeast');
grid on;

% 基音周期细节
subplot(3,2,6);
detail_range = search_min:min(search_max, length(voiced_autocorr));
time_detail = time_lags(detail_range);
plot(time_detail, voiced_autocorr(detail_range), 'b-', 'LineWidth', 2);
hold on;
plot(time_lags(pitch_period), voiced_autocorr(pitch_period), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title(sprintf('3.4.6 基音周期细节: %.1f Hz', pitch_frequency));
xlabel('延迟 (ms)'); ylabel('归一化自相关');
grid on;

%% 3.5 短时傅里叶分析
fprintf('\n=== 3.5 短时傅里叶分析 ===\n');

% FFT点数
nfft = 1024;
freq_axis = (0:nfft/2) * Fs / nfft;%正频率轴

% 计算频谱
voiced_fft = fft(voiced_data .* hamming(length(voiced_data))', nfft);%加窗FFT，减少频谱泄漏
unvoiced_fft = fft(unvoiced_data .* hamming(length(unvoiced_data))', nfft);

voiced_mag = abs(voiced_fft(1:nfft/2+1));% 幅度谱：显示各频率成分的强度
unvoiced_mag = abs(unvoiced_fft(1:nfft/2+1));
voiced_phase = angle(voiced_fft(1:nfft/2+1));% 相位谱：显示各频率成分的相位关系
unvoiced_phase = angle(unvoiced_fft(1:nfft/2+1));

% dB谱
voiced_mag_db = 20 * log10(voiced_mag + eps);% dB谱：对数尺度，更符合人耳感知
unvoiced_mag_db = 20 * log10(unvoiced_mag + eps);

% 共振峰提取函数


% 提取共振峰：共振峰是声道谐振产生的频谱峰值，对应声道的共振频率
lpc_order = 12;
formants = find_formants_lpc(voiced_data, Fs, lpc_order);

fprintf('前三个共振峰频率:\n');
for i = 1:min(3, length(formants.freqs))
    fprintf('  F%d: %.1f Hz\n', i, formants.freqs(i));
end

% 可视化傅里叶分析
figure('Position', [1, 1, 1920, 1080], 'Name', '傅里叶分析', 'NumberTitle', 'on');

% 时域波形
subplot(3,4,1);
plot(voiced_data, 'b-', 'LineWidth', 1.5);
title('3.5.1 浊音帧时域波形');
xlabel('采样点'); ylabel('幅度');
grid on;

subplot(3,4,2);
plot(unvoiced_data, 'r-', 'LineWidth', 1.5);
title('3.5.2 清音帧时域波形');
xlabel('采样点'); ylabel('幅度');
grid on;

% 幅度谱（线性）
subplot(3,4,3);
plot(freq_axis, voiced_mag, 'b-', 'LineWidth', 1.5);
title('3.5.3 浊音帧幅度谱（线性）');
xlabel('频率 (Hz)'); ylabel('幅度');
xlim([0 4000]); grid on;

subplot(3,4,4);
plot(freq_axis, unvoiced_mag, 'r-', 'LineWidth', 1.5);
title('3.5.4 清音帧幅度谱（线性）');
xlabel('频率 (Hz)'); ylabel('幅度');
xlim([0 4000]); grid on;

% 幅度谱（dB）
subplot(3,4,5);
plot(freq_axis, voiced_mag_db, 'b-', 'LineWidth', 1.5);
title('3.5.5 浊音帧幅度谱（dB）');
xlabel('频率 (Hz)'); ylabel('幅度 (dB)');
xlim([0 4000]); grid on;

subplot(3,4,6);
plot(freq_axis, unvoiced_mag_db, 'r-', 'LineWidth', 1.5);
title('3.5.6 清音帧幅度谱（dB）');
xlabel('频率 (Hz)'); ylabel('幅度 (dB)');
xlim([0 4000]); grid on;

% 相位谱
subplot(3,4,7);
plot(freq_axis, voiced_phase, 'b-', 'LineWidth', 1);
title('3.5.7 浊音帧相位谱');
xlabel('频率 (Hz)'); ylabel('相位 (rad)');
xlim([0 4000]); grid on;

subplot(3,4,8);
plot(freq_axis, unvoiced_phase, 'r-', 'LineWidth', 1);
title('3.5.8 清音帧相位谱');
xlabel('频率 (Hz)'); ylabel('相位 (rad)');
xlim([0 4000]); grid on;

% 频谱包络和共振峰
subplot(3,4,9);
plot(freq_axis, voiced_mag_db, 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
hold on;
plot(formants.freq_axis, formants.spectrum, 'b-', 'LineWidth', 2);
for i = 1:min(3, length(formants.freqs))
    [~, idx] = min(abs(formants.freq_axis - formants.freqs(i)));
    plot(formants.freqs(i), formants.spectrum(idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    text(formants.freqs(i), formants.spectrum(idx)+3, sprintf('F%d', i), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end
title('3.5.9 浊音帧频谱包络和共振峰');
xlabel('频率 (Hz)'); ylabel('幅度 (dB)');
xlim([0 4000]); legend('原始谱', '包络', '共振峰'); grid on;

% 频谱对比
subplot(3,4,10);
plot(freq_axis, voiced_mag_db, 'b-', 'LineWidth', 1.5);
hold on;
plot(freq_axis, unvoiced_mag_db, 'r-', 'LineWidth', 1.5);
title('3.5.10 浊音 vs 清音频谱对比');
xlabel('频率 (Hz)'); ylabel('幅度 (dB)');
xlim([0 4000]); legend('浊音', '清音'); grid on;

% 语谱图
subplot(3,4,11);
spectrogram(voiced_data, 256, 128, 1024, Fs, 'yaxis');
title('3.5.11 浊音帧语谱图');
colorbar;

subplot(3,4,12);
spectrogram(unvoiced_data, 256, 128, 1024, Fs, 'yaxis');
title('3.5.12 清音帧语谱图');
colorbar;

%% 总结输出
fprintf('\n=== 分析结果总结 ===\n');
fprintf('基音频率: %.1f Hz\n', pitch_frequency);
fprintf('共振峰频率: ');
for i = 1:min(3, length(formants.freqs))
    fprintf('F%d=%.1fHz ', i, formants.freqs(i));
end
fprintf('\n');


function corr = short_time_autocorr(signal, max_lag)
    N = length(signal);
    corr = zeros(1, max_lag);
    for lag = 0:max_lag-1
        corr(lag+1) = sum(signal(1:N-lag) .* signal(lag+1:N));
    end
end

function formants = find_formants_lpc(signal, Fs, order)
    % 预加重
    preemph = filter([1 -0.97], 1, signal);
    % LPC分析
    a = lpc(preemph, order);
    % 频率响应
    [h, f] = freqz(1, a, 1024, Fs);
    h_db = 20 * log10(abs(h));
    % 寻找峰值
    [peaks, locs] = findpeaks(h_db, 'SortStr', 'descend', 'NPeaks', 5);
    formant_freqs = f(locs);
    % 过滤合理范围
    valid_idx = formant_freqs > 150 & formant_freqs < 4000;
    formants.freqs = formant_freqs(valid_idx);
    formants.mags = peaks(valid_idx);
    formants.spectrum = h_db;
    formants.freq_axis = f;
end