%% Author: Dominic Alessi
%% Speech Recognition using MFCC and DTW
% Based on: Mohan & Babu (2014)

clear all;
close all;
clc;

% Configuration Parameters, sampling at 48 kHz and 25 ms long frames
config.fs = 48000;             % Native sample rate from phone
config.frame_length = 1200;    % 25ms at 48kHz
config.frame_overlap = 600;    % 12.5ms overlap
config.n_mfcc = 12;            % Number of MFCC coefficients - from paper
config.n_filters = 26;         % Number of Mel filter banks - from paper
config.pre_emphasis = 0.85;    % Pre-emphasis coefficient (k, not used since it over attenuates the audio signal)
config.dtw_threshold = 800;    % DTW distance threshold (may need adjustment)

% Define vocabulary
vocabulary = {'forward', 'left', 'right', 'reverse', 'control', 'start', 'stop', 'down', 'up', 'backward'}; % 10 word vocab
n_templates_per_word = 10; % Number of training samples per word


fprintf('TRAINING PHASE\n');
templates = struct();

for word_idx = 1:length(vocabulary)
    word = vocabulary{word_idx};

    % change n_templates per word if word == left or right here
    if strcmp('right', word)
        n_templates_per_word = 15;
    else
        n_templates_per_word = 10;
    end 
    fprintf('Processing templates for word: %s\n', word);
    
    templates.(word) = cell(n_templates_per_word, 1);
    
    for template_idx = 1:n_templates_per_word
        % Load from .m4a file
        filename = sprintf('training/%s_%d.m4a', word, template_idx);
        fprintf('  Loading: %s\n', filename);
        
        try
            [audio, fs_file] = audioread(filename);
            fprintf('    Original sample rate: %d Hz, Duration: %.2f sec\n', ...
                    fs_file, length(audio)/fs_file);
            
            % Resample if necessary
            if fs_file ~= config.fs
                fprintf('    Resampling from %d Hz to %d Hz...\n', fs_file, config.fs);
                audio = resample(audio, config.fs, fs_file);
            end
        catch ME
            error('Could not load file: %s\nMake sure file exists!\nError: %s', ...
                  filename, ME.message);
        end
        
        % Extract MFCC features
        mfcc_features = extract_mfcc(audio, config, word, template_idx);
        
        % Store template
        templates.(word){template_idx} = mfcc_features;
    end
    fprintf('\n');
end

% Save templates
save('speech_templates.mat', 'templates', 'vocabulary', 'config');
fprintf('Templates saved to speech_templates.mat\n\n');

%% TESTING PHASE

fprintf('TESTING PHASE\n');

while true
    fprintf('\nEnter test filename (e.g., test_forward_1.m4a) or "quit" to exit: ');
    filename = input('', 's');
    
    if strcmpi(filename, 'quit')
        break;
    end
    
    % Load test file
    filepath = fullfile('test', filename);
    fprintf('Loading: %s\n', filepath);
    
    try
        [test_audio, fs_file] = audioread(filepath);
        fprintf('Sample rate: %d Hz, Duration: %.2f sec\n', ...
                fs_file, length(test_audio)/fs_file);
        
        % Resample if necessary
        if fs_file ~= config.fs
            fprintf('Resampling from %d Hz to %d Hz...\n', fs_file, config.fs);
            test_audio = resample(test_audio, config.fs, fs_file);
        end
    catch ME
        fprintf('Error loading file: %s\nTry again.\n', ME.message);
        continue;
    end
    
    % Extract MFCC features from test sample
    test_features = extract_mfcc(test_audio, config, '', 0);
    fprintf('Test features: %d features x %d frames\n', ...
            size(test_features, 1), size(test_features, 2));
    
    % Compare with all templates using DTW
    min_distance = inf;
    recognized_word = '';
    all_distances = zeros(length(vocabulary), 1);
    
    for word_idx = 1:length(vocabulary)
        word = vocabulary{word_idx};
        word_templates = templates.(word);
        
        min_word_distance = inf;
        for template_idx = 1:length(word_templates)
            template_features = word_templates{template_idx};
            
            % Calculate DTW distance
            distance = dtw_distance(test_features, template_features);
            
            if distance < min_word_distance
                min_word_distance = distance;
            end
        end
        
        all_distances(word_idx) = min_word_distance;
        
        if min_word_distance < min_distance
            min_distance = min_word_distance;
            recognized_word = word;
        end
    end
    
    % Display results
    fprintf('\nRecognition Results\n');
    fprintf('Recognized word: %s\n', upper(recognized_word));
    fprintf('DTW distance: %.2f\n', min_distance);
    
    fprintf('\nAll distances:\n');
    for i = 1:length(vocabulary)
        fprintf('  %s: %.2f', vocabulary{i}, all_distances(i));
        if strcmpi(vocabulary{i}, recognized_word)
            fprintf(' <-- MATCH');
        end
        fprintf('\n');
    end
    
    if min_distance < config.dtw_threshold
        fprintf('\nConfidence: HIGH (distance < threshold)\n');
    else
        fprintf('\nConfidence: LOW (distance >= threshold)\n');
        fprintf('Word may not be in vocabulary!\n');
    end
    fprintf('---------------------------\n\n');
end

fprintf('Program terminated.\n');

function mfcc_features = extract_mfcc(audio, config, word, template_idx) % this entire function is called once for every audio
% file used by the system. Thus we need to keep track of the word and it's
% index.
    
    %% Preprocessing
    % Convert to mono if stereo, unlikely any of the audio is stereo, but
    % good safety net
    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end
    
    % Ensure column vector
    audio = audio(:);
    
    % Scale to 16-bit range for better numerical stability
    if max(abs(audio)) <= 1
        audio = audio * 32768;  % Scale to 16-bit signed integer range
    end

    audioBeforeZeroAlignment = audio; % note that the naming of this variable is misleading. It it prior to any preprocessing:
% pre-empasis filtering, noise gate, and zero allignment.
    
%% Filtering
    % Design bandpass filter: 50 Hz - 8000 Hz
    % This replaces the simple pre-emphasis filter
    low_freq = 50;      % High-pass cutoff (remove DC and rumble)
    high_freq = 8000;   % Low-pass cutoff (speech content ends here)
    
    % Design 6th-order Butterworth bandpass filter
    [b, a] = butter(6, [low_freq, high_freq]/(config.fs/2), 'bandpass');
    
    % Apply filter
    audio = filtfilt(b, a, audio);  % Zero-phase filtering (no delay)

    % pre-emphasis filter (HPF):
    %audio = filter([1, -config.pre_emphasis], 1, audio);
    % Commented out because it over attuates audio signal

%% Noise gate - remove samples below threshold
    noise_threshold = 0.01 * max(abs(audio));
    audio(abs(audio) < noise_threshold) = 0;

    % Zero-alignment - find first non-zero sample and trim
    first_sample = find(abs(audio) > noise_threshold, 1, 'first');
    if ~isempty(first_sample)
        audio = audio(first_sample:end);
    end

    audio = enforce_fixed_duration(audio, 2.0, config.fs);

    %% Plot audio before and after zero-alignment (only for left_1.m4a)
    if strcmp(word, 'control') && template_idx == 1

       figure('Position', [100, 100, 1000, 600]);
        
        % Calculate PSDs
        [pxx_before, f_before] = pwelch(audioBeforeZeroAlignment, ...
                                         hamming(2048), 1024, 2048, config.fs);
        [pxx_after, f_after] = pwelch(audio, ...
                                       hamming(2048), 1024, 2048, config.fs);
        
        % Plot both on same axes
        plot(f_before/1000, 10*log10(pxx_before), 'b', 'LineWidth', 2, ...
             'DisplayName', 'Before Filter');
        hold on;
        plot(f_after/1000, 10*log10(pxx_after), 'r', 'LineWidth', 2, ...
             'DisplayName', 'After Bandpass (50-8000 Hz)');
      
        
        grid on;
        xlabel('Frequency (kHz)', 'FontSize', 14);
        ylabel('Power/Frequency (dB/Hz)', 'FontSize', 14);
        title('PSD for word "control": Effect of Bandpass Filtering', ...
              'FontSize', 16, 'FontWeight', 'bold');
        legend('Location', 'northeast', 'FontSize', 12);
        xlim([0 config.fs/2000]);
        ylim([-80 40]);

         % Add vertical lines showing filter cutoffs
        xline(0.05, 'g--', 'LineWidth', 1.5, 'Label', '50 Hz', ...
              'LabelHorizontalAlignment', 'left');
        xline(8, 'g--', 'LineWidth', 1.5, 'Label', '8 kHz', ...
              'LabelHorizontalAlignment', 'right');
        
        
        text(10, 30, sprintf('Sample Rate: %d kHz\nWord: left\nSample: 1', ...
             config.fs/1000), 'FontSize', 11, 'BackgroundColor', 'white');

        figure('Position', [100, 100, 1200, 600]);
        
        % Time axis (in seconds)
        time_before = (0:length(audioBeforeZeroAlignment)-1) / config.fs;
        time_after = (0:length(audio)-1) / config.fs;
        
        % Normalize back to original scale (divide by 32768)
        audio_before_normalized = audioBeforeZeroAlignment / 32768;
        audio_after_normalized = audio / 32768;
        
        % Plot before zero-alignment
        subplot(2, 1, 1);
        plot(time_before, audio_before_normalized);
        xlabel('Time (seconds)');
        ylabel('Amplitude');
        title('Before Zero-Alignment (48 kHz)');
        grid on;
        
        % Plot after zero-alignment
        subplot(2, 1, 2);
        plot(time_after, audio_after_normalized);
        xlabel('Time (seconds)');
        ylabel('Amplitude');
        title('After Zero-Alignment (48 kHz)');
        grid on;
    end
    
    %% Framing and Windowing
    frame_step = config.frame_length - config.frame_overlap; % 1200 - 600 = 600
    n_frames = floor((length(audio) - config.frame_length) / frame_step) + 1;
    
    % Initialize frame matrix
    frames = zeros(config.frame_length, n_frames);
    
    % Extract frames
    for i = 1:n_frames
        start_idx = (i-1) * frame_step + 1;
        end_idx = min(start_idx + config.frame_length - 1, length(audio));
        
        if end_idx - start_idx + 1 < config.frame_length
            % Pad last frame if needed
            frame_data = [audio(start_idx:end_idx); ...
                         zeros(config.frame_length - (end_idx - start_idx + 1), 1)];
            frames(:, i) = frame_data;
        else
            frames(:, i) = audio(start_idx:end_idx);
        end
    end
    
    % Apply Hamming window to each frame
    hamming_window = hamming(config.frame_length);
    for i = 1:n_frames
        frames(:, i) = frames(:, i) .* hamming_window;
    end
    
    %% FFT
    nfft = 2048;  % Increased from 512 for 48kHz (better frequency resolution)
    fft_frames = fft(frames, nfft); % Since each column is a frame, we take the FFT of each column in the 2D matrix
    fft_frames = abs(fft_frames(1:nfft/2+1, :));
    power_spectrum = (fft_frames .^ 2) / nfft;
    
    %% Mel Filter Bank
    mel_filterbank = create_mel_filterbank(config.n_filters, nfft, config.fs, 0, 8000); 
    filter_bank_energy = mel_filterbank * power_spectrum;
    log_filter_bank_energy = log(filter_bank_energy + eps);

    if strcmp(word, 'control') && template_idx == 1
        plotMelFilters();
    end
    
    %% DCT
    mfcc_coeffs = dct(log_filter_bank_energy);
    
    if strcmp(word, 'left') && template_idx == 1
        % Plot all 26 coefficients for one frame
        figure;
        stem(0:25, abs(mfcc_coeffs(:, 10)), 'filled', 'LineWidth', 2);
        xlabel('MFCC Coefficient Index', 'FontSize', 12);
        ylabel('Magnitude', 'FontSize', 12);
        title('MFCC Coefficient Magnitudes for the left_1.m4a', 'FontSize', 14);
        grid on;
        
        % Add vertical line at coefficient 12
        xline(12.5, 'r--', 'LineWidth', 2, 'Label', 'Cutoff at 12');
        
        % Annotate
        text(5, max(abs(mfcc_coeffs(:, 10)))*0.9, 'High energy', 'FontSize', 11);
        text(18, max(abs(mfcc_coeffs(:, 10)))*0.3, 'Low energy (noise)', 'FontSize', 11);
    end

    % Keep only the first n_mfcc coefficients (12 from paper)
    mfcc_coeffs = mfcc_coeffs(1:config.n_mfcc, :);
    
    
    %% Delta and Delta-Delta Coefficients
    delta_coeffs = calculate_delta(mfcc_coeffs, 2);
    delta_delta_coeffs = calculate_delta(delta_coeffs, 2);
    
    %% Combine all features (36 total: 12 + 12 + 12)
    mfcc_features = [mfcc_coeffs; delta_coeffs; delta_delta_coeffs];

   % feature normalization
    for i = 1:size(mfcc_features, 1)
        feat_std = std(mfcc_features(i, :));
        if feat_std > eps
            mfcc_features(i, :) = mfcc_features(i, :) / feat_std;
        end
    end

end

function mel_filterbank = create_mel_filterbank(n_filters, nfft, fs, fmin, fmax)
    % CREATE_MEL_FILTERBANK Create triangular Mel-spaced filter bank
    % NEW: fmin, fmax parameters to limit frequency range
    
    low_freq = fmin;    % NEW: use parameter instead of 0
    high_freq = fmax;   % NEW: use parameter instead of fs/2
    
    % Rest of code stays the same...
    low_mel = hz_to_mel(low_freq);
    high_mel = hz_to_mel(high_freq);
    
    mel_points = linspace(low_mel, high_mel, n_filters + 2);
    hz_points = mel_to_hz(mel_points);
    bin_points = floor((nfft + 1) * hz_points / fs);
    
    mel_filterbank = zeros(n_filters, nfft/2 + 1);
    
    for i = 1:n_filters
        left = bin_points(i);
        center = bin_points(i + 1);
        right = bin_points(i + 2);
        
        % Rising slope
        for k = left:center
            mel_filterbank(i, k+1) = (k - left) / (center - left);
        end
        
        % Falling slope
        for k = center:right
            mel_filterbank(i, k+1) = (right - k) / (right - center);
        end
    end
end

function mel = hz_to_mel(hz)
    % Convert frequency in Hz to Mel scale
    % Using formula from paper: mel = 2595 * log10(1 + hz/700)
    mel = 2595 * log10(1 + hz / 700);
end

function hz = mel_to_hz(mel)
    % Convert Mel scale to frequency in Hz
    hz = 700 * (10 .^ (mel / 2595) - 1);
end

function delta = calculate_delta(coeffs, M)
    % CALCULATE_DELTA Calculate delta (derivative) coefficients
    
    [n_coeffs, n_frames] = size(coeffs);
    delta = zeros(size(coeffs));
    
    for t = 1:n_frames
        t_minus_M = max(1, t - M);
        t_plus_M = min(n_frames, t + M);
        delta(:, t) = coeffs(:, t_plus_M) - coeffs(:, t_minus_M);
    end
end

function distance = dtw_distance(seq1, seq2)
    % DTW_DISTANCE Calculate Dynamic Time Warping distance
    
    [n_features1, n_frames1] = size(seq1);
    [n_features2, n_frames2] = size(seq2);
    
    if n_features1 ~= n_features2
        error('Feature dimensions must match!');
    end
    
    % Initialize DTW cost matrix
    dtw_matrix = inf(n_frames1, n_frames2);
    
    %% Calculate local distance matrix
    local_distance = zeros(n_frames1, n_frames2);
    
    for i = 1:n_frames1
        for j = 1:n_frames2
            % Euclidean distance between feature vectors
            local_distance(i, j) = sqrt(sum((seq1(:, i) - seq2(:, j)) .^ 2));
        end
    end
    

    dtw_matrix(1, 1) = local_distance(1, 1);
    
    % Initialize first column
    for i = 2:n_frames1
        dtw_matrix(i, 1) = dtw_matrix(i-1, 1) + local_distance(i, 1);
    end
    
    % Initialize first row
    for j = 2:n_frames2
        dtw_matrix(1, j) = dtw_matrix(1, j-1) + local_distance(1, j);
    end
    
    % Fill in the rest
    for i = 2:n_frames1
        for j = 2:n_frames2
            min_previous = min([dtw_matrix(i-1, j-1), ...
                               dtw_matrix(i-1, j), ...
                               dtw_matrix(i, j-1)]);
            dtw_matrix(i, j) = local_distance(i, j) + min_previous;
        end
    end
    
    % Normalize by path length
    distance = dtw_matrix(n_frames1, n_frames2);
end

function plotMelFilters = plotMelFilters()
    %% Plot Mel Filter Banks
    % Call this after training or add to your main script
    
    % Create the filterbank
    n_filters = 26;
    nfft = 2048;
    fs = 48000;
    
    mel_filterbank = create_mel_filterbank(n_filters, nfft, fs, 0, 8000);
    
    % Create frequency axis (in Hz)
    freq_axis = linspace(0, fs/2, nfft/2 + 1);
    
    % Plot first 12 filters
    figure('Position', [100, 100, 1200, 600]);
    
    % Plot each filter
    for i = 1:26
        plot(freq_axis/1000, mel_filterbank(i, :), 'LineWidth', 2);
        hold on;
    end
    
    % Formatting
    grid on;
    xlabel('Frequency (kHz)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Filter Amplitude', 'FontSize', 14, 'FontWeight', 'bold');
    title('Mel-Spaced Triangular Filter Banks', ...
          'FontSize', 16, 'FontWeight', 'bold');
    
    xlim([0 9]);  % Focus on speech range (0-8 kHz)
    ylim([0 1.1]);
    
    % Add grid for readability
    grid minor;
    
    
    hold off;
end

function audio = enforce_fixed_duration(audio, target_duration_sec, fs)
    % ENFORCE_FIXED_DURATION Pad or trim audio to exact duration
    %
    % Inputs:
    %   audio - audio signal (column vector)
    %   target_duration_sec - desired duration in seconds (e.g., 2.0)
    %   fs - sample rate in Hz (e.g., 48000)
    %
    % Output:
    %   audio - fixed-length audio signal
    
    target_samples = round(target_duration_sec * fs);
    current_samples = length(audio);
    
    if current_samples < target_samples
        % Pad with zeros at the end
        padding = zeros(target_samples - current_samples, 1);
        audio = [audio; padding];
        fprintf('    Padded %d samples (%.2f sec)\n', ...
                target_samples - current_samples, ...
                (target_samples - current_samples) / fs);
    elseif current_samples > target_samples
        % Trim excess samples
        audio = audio(1:target_samples);
        fprintf('    Trimmed %d samples (%.2f sec)\n', ...
                current_samples - target_samples, ...
                (current_samples - target_samples) / fs);
    else
        fprintf('    Already exact duration (%.2f sec)\n', target_duration_sec);
    end
end



