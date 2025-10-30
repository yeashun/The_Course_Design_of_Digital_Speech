clear all; close all; clc;

%% 加载语音数据
[speech, Fs] = audioread('yeshun.wav');
x = speech(10000:30000);

%% 分帧处理
nwin = round(0.03 * Fs);  % 帧长30ms
ninc = round(0.01 * Fs);  % 帧移10ms
nf = fix((length(x) - nwin + ninc) / ninc);

frame = zeros(nf, nwin);
for i = 1:nf
    start_index = (i-1) * ninc + 1;
    end_index = start_index + nwin - 1;
    if end_index <= length(x)
        frame(i, :) = x(start_index:end_index);
    end
end
w = hamming(nwin)';
frame_windowed = frame .* w;

%% 浊音/清音判断
shortenergy = sum(frame_windowed.^2, 2);
nzcr = zeros(nf, 1);
for i = 1:nf
    current_frame = frame_windowed(i, :);
    for j = 1:nwin-1
        if current_frame(j) * current_frame(j+1) < 0
            nzcr(i) = nzcr(i) + 1;
        end
    end
end

energy_threshold = 0.1 * max(shortenergy);
zcr_threshold = 0.3 * max(nzcr);
voice_decision = zeros(nf, 1);
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

%% 选取浊音帧和清音帧
voiced_frames = find(voice_decision == 1);
unvoiced_frames = find(voice_decision == 0);

if isempty(voiced_frames)
    voiced_frame_idx = min(50, nf);
else
    voiced_frame_idx = voiced_frames(ceil(length(voiced_frames)/2));
end

if isempty(unvoiced_frames)
    unvoiced_frame_idx = min(20, nf);
else
    unvoiced_frame_idx = unvoiced_frames(ceil(length(unvoiced_frames)/2));
end

fprintf('浊音帧索引: %d\n', voiced_frame_idx);
fprintf('清音帧索引: %d\n', unvoiced_frame_idx);

voiced_data = frame_windowed(voiced_frame_idx, :);
unvoiced_data = frame_windowed(unvoiced_frame_idx, :);

%% 3.1 线性预测分析（P=18阶）
fprintf('\n=== 3.1 线性预测分析（P=18阶） ===\n');

% 分析浊音帧
x1_voiced = voiced_data;
L_voiced = length(x1_voiced);
p_voiced = 18;  % 固定为18阶
[a_voiced, g_voiced] = lpc(x1_voiced, p_voiced);
est_x_voiced = filter([0 -a_voiced(2:end)], 1, x1_voiced);
e_voiced = x1_voiced - est_x_voiced;

% 分析清音帧
x1_unvoiced = unvoiced_data;
L_unvoiced = length(x1_unvoiced);
p_unvoiced = 18;  % 固定为18阶
[a_unvoiced, g_unvoiced] = lpc(x1_unvoiced, p_unvoiced);
est_x_unvoiced = filter([0 -a_unvoiced(2:end)], 1, x1_unvoiced);
e_unvoiced = x1_unvoiced - est_x_unvoiced;

fprintf('浊音帧预测增益: %.4f\n', g_voiced);
fprintf('清音帧预测增益: %.4f\n', g_unvoiced);

%% 3.2 显示原始语音、预测语音和预测误差波形
fprintf('\n=== 3.2 波形显示和误差分析 ===\n');

% 浊音帧分析
% figure('Position', [100, 100, 1200, 800]);
figure('Position', [1, 1, 1920, 1080], 'Name', '浊音帧分析', 'NumberTitle', 'on');
subplot(2,3,1);
t_voiced = (0:L_voiced-1)/Fs;
plot(t_voiced, x1_voiced, 'k', 'LineWidth', 1.5);
title('3.2.1 浊音帧原始语音');
xlabel('时间/s'); ylabel('幅度');
legend('原始语音信号'); grid on;

subplot(2,3,2);
plot(t_voiced, est_x_voiced, 'b', 'LineWidth', 1.5);
title('3.2.2 浊音帧预测信号');
xlabel('时间/s'); ylabel('幅度');
legend('预测信号'); grid on;

subplot(2,3,3);
plot(t_voiced, e_voiced, 'r', 'LineWidth', 1.5);
title('3.2.3 浊音帧预测误差');
xlabel('时间/s'); ylabel('幅度');
legend('预测误差'); grid on;

% 浊音帧自相关分析
subplot(2,3,4);
yxcorr_voiced = xcorr(x1_voiced, 'coeff');
plot(0:L_voiced-1, yxcorr_voiced(L_voiced:2*L_voiced-1), 'k', 'LineWidth', 1.5);
title('3.2.4 浊音帧原始语音自相关');
xlabel('样点序号/n'); ylabel('幅度');
legend('原始语音的自相关'); grid on;

subplot(2,3,5);
[excorr_voiced, ~] = xcorr(e_voiced, 'coeff');
plot(0:L_voiced-1, excorr_voiced(L_voiced:2*L_voiced-1), 'r', 'LineWidth', 1.5);
title('3.2.5 浊音帧预测误差自相关');
xlabel('样点序号/n'); ylabel('幅度');
legend('预测误差的自相关'); grid on;

% 预测系数显示
subplot(2,3,6);
stem(1:p_voiced, a_voiced(2:end), 'b', 'LineWidth', 1.5, 'Marker', 'o');
title('3.2.6 浊音帧LPC系数');
xlabel('系数序号'); ylabel('系数值');
grid on;

% 清音帧分析
% figure('Position', [100, 100, 1200, 800]);
figure('Position', [1, 1, 1920, 1080], 'Name', '清音帧分析', 'NumberTitle', 'on');
subplot(2,3,1);
t_unvoiced = (0:L_unvoiced-1)/Fs;
plot(t_unvoiced, x1_unvoiced, 'k', 'LineWidth', 1.5);
title('3.2.7 清音帧原始语音');
xlabel('时间/s'); ylabel('幅度');
legend('原始语音信号'); grid on;

subplot(2,3,2);
plot(t_unvoiced, est_x_unvoiced, 'b', 'LineWidth', 1.5);
title('3.2.8 清音帧预测信号');
xlabel('时间/s'); ylabel('幅度');
legend('预测信号'); grid on;

subplot(2,3,3);
plot(t_unvoiced, e_unvoiced, 'r', 'LineWidth', 1.5);
title('3.2.9 清音帧预测误差');
xlabel('时间/s'); ylabel('幅度');
legend('预测误差'); grid on;

% 清音帧自相关分析
subplot(2,3,4);
yxcorr_unvoiced = xcorr(x1_unvoiced, 'coeff');
plot(0:L_unvoiced-1, yxcorr_unvoiced(L_unvoiced:2*L_unvoiced-1), 'k', 'LineWidth', 1.5);
title('3.2.10 清音帧原始语音自相关');
xlabel('样点序号/n'); ylabel('幅度');
legend('原始语音的自相关'); grid on;

subplot(2,3,5);
[excorr_unvoiced, ~] = xcorr(e_unvoiced, 'coeff');
plot(0:L_unvoiced-1, excorr_unvoiced(L_unvoiced:2*L_unvoiced-1), 'r', 'LineWidth', 1.5);
title('3.2.11 清音帧预测误差自相关');
xlabel('样点序号/n'); ylabel('幅度');
legend('预测误差的自相关'); grid on;

% 预测系数显示
subplot(2,3,6);
stem(1:p_unvoiced, a_unvoiced(2:end), 'b', 'LineWidth', 1.5, 'Marker', 'o');
title('3.2.12 清音帧LPC系数');
xlabel('系数序号'); ylabel('系数值');
grid on;

%% 3.3 预测系数谱分析
fprintf('\n=== 3.3 预测系数谱分析 ===\n');

Nfft = 1024;
freq_axis = (1:Nfft/2) * Fs / Nfft;

% 浊音帧谱分析
yp_voiced = abs(fft(a_voiced, Nfft));
Yp_voiced = 20*log10(sqrt(g_voiced) * p_voiced ./ yp_voiced);
yd_voiced = abs(fft(x1_voiced, Nfft));
Yd_voiced = 20*log10(yd_voiced);

% 清音帧谱分析
yp_unvoiced = abs(fft(a_unvoiced, Nfft));
Yp_unvoiced = 20*log10(sqrt(g_unvoiced) * p_unvoiced ./ yp_unvoiced);
yd_unvoiced = abs(fft(x1_unvoiced, Nfft));
Yd_unvoiced = 20*log10(yd_unvoiced);

% 浊音帧频谱对比
% figure('Position', [100, 100, 1200, 800]);
figure('Position', [1, 1, 1920, 1080], 'Name', '浊音帧频谱对比', 'NumberTitle', 'on');
subplot(2,2,1);
plot(freq_axis, Yp_voiced(1:Nfft/2), '--r', 'LineWidth', 2);
hold on;
plot(freq_axis, Yd_voiced(1:Nfft/2), 'k', 'LineWidth', 1);
hold off;
title('3.3.1 浊音帧频谱对比');
xlabel('频率/Hz'); ylabel('幅度/dB');
legend('线性预测谱包络', '语音短时谱');
xlim([0 4000]); grid on;

% 清音帧频谱对比
subplot(2,2,2);
plot(freq_axis, Yp_unvoiced(1:Nfft/2), '--r', 'LineWidth', 2);
hold on;
plot(freq_axis, Yd_unvoiced(1:Nfft/2), 'k', 'LineWidth', 1);
hold off;
title('3.3.2 清音帧频谱对比');
xlabel('频率/Hz'); ylabel('幅度/dB');
legend('线性预测谱包络', '语音短时谱');
xlim([0 4000]); grid on;

% LPC谱对比
subplot(2,2,3);
plot(freq_axis, Yp_voiced(1:Nfft/2), 'b', 'LineWidth', 2);
hold on;
plot(freq_axis, Yp_unvoiced(1:Nfft/2), 'r', 'LineWidth', 2);
hold off;
title('3.3.3 浊音 vs 清音LPC谱对比');
xlabel('频率/Hz'); ylabel('幅度/dB');
legend('浊音LPC谱', '清音LPC谱');
xlim([0 4000]); grid on;

% 复倒谱包络分析（浊音帧）
subplot(2,2,4);
h_voiced = zeros(1, Nfft);
h_voiced(1) = -a_voiced(2);
for n = 2:Nfft
    if n < p_voiced + 1
        h_voiced(n) = -a_voiced(n+1);
        for k = 1:n-1
            h_voiced(n) = h_voiced(n) - (1-k)/n * a_voiced(k+1) * h_voiced(n-k);
        end
    else
        h_voiced(n) = 0;
        for k = 1:p_voiced
            h_voiced(n) = h_voiced(n) - (1-k)/n * a_voiced(k+1) * h_voiced(n-k);
        end
    end
end
Hl_voiced = fft(h_voiced, Nfft);

plot(freq_axis, 20*log10(sqrt(g_voiced) * p_voiced ./ abs(Hl_voiced(1:Nfft/2))), '--g', 'LineWidth', 2);
hold on;
plot(freq_axis, Yp_voiced(1:Nfft/2), '--r', 'LineWidth', 1);
plot(freq_axis, Yd_voiced(1:Nfft/2), 'k', 'LineWidth', 0.5);
hold off;
title('3.3.4 浊音帧复倒谱包络分析');
xlabel('频率/Hz'); ylabel('幅度/dB');
legend('LPC复倒谱包络', '线性预测谱包络', '语音短时谱');
xlim([0 4000]); grid on;

%% 误差统计分析
fprintf('\n=== 预测误差统计分析 ===\n');

% 计算统计量
error_stats = zeros(4,2);
error_stats(1,1) = std(e_voiced);          % 浊音误差标准差
error_stats(1,2) = std(e_unvoiced);        % 清音误差标准差
error_stats(2,1) = mean(abs(e_voiced));    % 浊音误差平均绝对值
error_stats(2,2) = mean(abs(e_unvoiced));  % 清音误差平均绝对值
error_stats(3,1) = var(e_voiced);          % 浊音误差方差
error_stats(3,2) = var(e_unvoiced);        % 清音误差方差
error_stats(4,1) = 10*log10(var(x1_voiced)/var(e_voiced));  % 浊音信噪比
error_stats(4,2) = 10*log10(var(x1_unvoiced)/var(e_unvoiced));  % 清音信噪比

% figure('Position', [100, 100, 800, 600]);
figure('Position', [1, 1, 1920, 1080], 'Name', '预测误差统计对比', 'NumberTitle', 'on');
subplot(2,1,1);
bar(error_stats(1:3,:)');
title('预测误差统计对比');
ylabel('数值');
legend('浊音帧', '清音帧', 'Location', 'northeast');
set(gca, 'XTickLabel', {'标准差', '平均绝对值', '方差'});
grid on;

subplot(2,1,2);
bar(error_stats(4,:));
title('预测信噪比对比');
ylabel('信噪比 (dB)');
set(gca, 'XTickLabel', {'浊音帧', '清音帧'});
grid on;


