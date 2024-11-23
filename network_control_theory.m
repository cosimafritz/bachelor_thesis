addpath('/Users/macbookcosima/Downloads/optim_fast');

load cocoConnectomes.mat;
load cocostates.mat;

% Extract relevant variables 
connectomes_data = connectomes.conn; % Connectivity matrix
participants = connectomes.participants; 
states_data = states.states; % States

% Number of participants in connectomes_data = 73
num_connectome_participants = size(connectomes_data, 4);

% Number of participants in states_data = 76
num_state_participants = size(states_data, 3);

% Determine the minimum number of participants to ensure matching data
num_participants = min(num_connectome_participants, num_state_participants);


% Loop over all participants
for subj_index = 1:num_participants
    % Select subject
    subj_id = participants(subj_index); % ID of the participant

    % Extract connectivity matrix for the current participant
    % FA-weighted networks at position 2
    A = connectomes_data(:, :, 2, subj_index);

    % Stabilized matrix A_star:
    A_star = A ./ (eigs(A, 1) + 1) - eye(size(A, 1));

    % Extract ICNs for the current participant
    con = states_data(:, 1, subj_index); % Cingulo-opercular Network
    dan = states_data(:, 2, subj_index); % Dorsal Attention Network
    dmn = states_data(:, 3, subj_index); % Default Mode Network
    ecn = states_data(:, 4, subj_index); % Executive Control Network
    fpn = states_data(:, 5, subj_index); % Frontoparietal Network
    smn = states_data(:, 6, subj_index); % Somatomotor Network
    visual1 = states_data(:, 7, subj_index); % Primary Visual Network
    visual2 = states_data(:, 8, subj_index); % Secondary Visual Network

    % ICN list
    icns = {con, dan, dmn, ecn, fpn, smn, visual1, visual2};
    icn_names = {'CON', 'DAN', 'DMN', 'ECN', 'FPN', 'SMN', 'VIS1', 'VIS2'};

    % Loop over all ICNs for current subject
    for icn_idx1 = 1:length(icns)
        for icn_idx2 = 1:length(icns)
            x0 = icns{icn_idx1};
            xf = icns{icn_idx2};

            % Parameter settings: 
            T = 1;  % time horizon 
            rho = 1; % penalty term 

            D = eye(size(A, 1)); % Set matrix D as identity
            S = eye(size(A, 1)); % Set matrix S as identity

            % Run optim_fun
            [X_opt, U_opt, n_err] = optim_fun(A_star, T, D, x0, xf, rho, S);

            % Control energy over time 
            control_over_time(:, subj_index, icn_idx1, icn_idx2) = mean(U_opt.^2, 2);

            % Control energy for transition x0 to xf
            ce = sum(trapz(U_opt.^2));
            control_energy(icn_idx1, icn_idx2, subj_index) = ce; % Store total energy for each ICN transition and each participant
        end
    end
end

%% Save results to .mat file
save('control_energy_results.mat', 'control_energy', 'control_over_time', 'participants', 'icn_names');


%% Visualizations
% outputDir = '/Users/macbookcosima/Desktop/Bachelorarbeit/Plots Control Energy';
addpath('/Users/macbookcosima/Downloads/ENIGMA-2.0.0/matlab/scripts/useful/export_fig-master');


%% Heatmap: average control energy for all transitions with control energy values

% average control energy across all participants
average_control_energy = mean(control_energy, 3);

figure;
heatmap(icn_names, icn_names, average_control_energy);
title('Average Control Energy for RSN Transitions');
xlabel('Target RSN');
ylabel('Initial RSN');
colorbar;

colormap(jet);

set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'Average_Control_Energy_Heatmap_All_Participants1.pdf'));


%% Histograms of control energy, 95th percentile

percentile_95 = prctile(control_energy(:), 95);

x_max = percentile_95;

figure;
colors = lines(length(icns));

num_bins = 50;
bin_width = x_max / num_bins;

max_freq = 0;
for icn_idx1 = 1:length(icns)
    for icn_idx2 = 1:length(icns)
        [counts, ~] = histcounts(squeeze(control_energy(icn_idx1, icn_idx2, :)), 'BinWidth', bin_width);
        max_freq = max(max_freq, max(counts));
    end
end
y_max = max_freq * 1.1;

for icn_idx1 = 1:length(icns)
    for icn_idx2 = 1:length(icns)
        subplot(length(icns), length(icns), (icn_idx1-1)*length(icns) + icn_idx2);
        histogram(squeeze(control_energy(icn_idx1, icn_idx2, :)), 'FaceColor', colors(icn_idx1, :), 'BinWidth', bin_width);
        title([icn_names{icn_idx1} '->' icn_names{icn_idx2}], 'FontSize', 6);
        xlabel('Control Energy', 'FontSize', 4);
        ylabel('Frequency', 'FontSize', 6);
        xlim([0, x_max]);
        ylim([0, y_max]);
        set(gca, 'FontSize', 6);
    end
end

sgtitle('Histogram of Control Energy (95th percentile) for RSN Transitions', 'FontSize', 8);

set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'Control_Energy_Histogram_95th_Percentile.pdf'));


%% Boxplot: control energy for all transitions

boxplot_data = {};
group_labels = {};

for icn_idx1 = 1:length(icn_names)
    for icn_idx2 = 1:length(icn_names)
        data = squeeze(control_energy(icn_idx1, icn_idx2, :));
        boxplot_data = [boxplot_data; num2cell(data)];
        
        transition_label = [icn_names{icn_idx1}, ' -> ', icn_names{icn_idx2}];
        group_labels = [group_labels; repmat({transition_label}, length(data), 1)];
    end
end

boxplot_data = cell2mat(boxplot_data);
group_labels = categorical(group_labels);

figure;
boxplot(boxplot_data, group_labels);
title('Control Energy for RSN Transitions');
ylabel('Control Energy');
xlabel('RSN Transitions');

set(gca, 'XTickLabelRotation', 90);
set(gca, 'FontSize', 6); 

set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'Control_Energy_Boxplot_All_Participants.pdf'));


%% Average Temporal Evolution of Control Energy for Each Initial State

average_control_over_time = mean(control_over_time, 2);

figure;
num_icns = length(icn_names);
for icn_idx1 = 1:num_icns
    subplot(3, 3, icn_idx1);
    hold on;
    for icn_idx2 = 1:num_icns
        avg_ce_ot = squeeze(average_control_over_time(:, :, icn_idx1, icn_idx2));
        plot(avg_ce_ot, 'LineWidth', 1.5, 'DisplayName', ['to ', icn_names{icn_idx2}]);
    end
    hold off;
    title(['Initial State: ', icn_names{icn_idx1}], 'FontSize', 8);
    xlabel('Time Steps', 'FontSize', 6);
    ylabel('Average Control Energy', 'FontSize', 6);
    legend('show', 'Location', 'northwest', 'FontSize', 4);
    grid on;
    set(gca, 'FontSize', 6);
    ylim([0 50]);
end

sgtitle('Average Temporal Evolution of Control Energy for Each Initial State', 'FontSize', 10);

set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'Average_Temporal_Evolution_Control_Energy_Subplots.pdf'));


%% Radar Chart: Average Control Energy

average_control_energy = mean(control_energy, 3);

num_icns = length(icn_names);
theta = linspace(0, 2*pi, num_icns + 1);

extended_data = [average_control_energy, average_control_energy(:, 1)];

figure;
for icn_idx1 = 1:num_icns
    subplot(3, 3, icn_idx1);
    polarplot(theta, extended_data(icn_idx1, :), 'LineWidth', 2);
    title(['Initial State: ', icn_names{icn_idx1}]);
    ax = gca;
    ax.ThetaTick = rad2deg(theta(1:end-1));
    ax.ThetaTickLabel = icn_names;
    ax.RLim = [min(average_control_energy(:)), max(average_control_energy(:))];
end

sgtitle('Radar Charts of Average Control Energy for Each Initial State');

set(gcf, 'Color', 'w');

export_fig(gcf,fullfile(outputDir, 'Radar_Charts_Average_Control_Energy.pdf'));


%% Violinplot Control Energy 

[n, m, p] = size(control_energy);

reshaped_control_energy = reshape(control_energy, [n * m, p]);

transition_labels = cell(n * m, 1);
idx = 1;
for i = 1:n
    for j = 1:m
        transition_labels{idx} = [icn_names{i}, ' -> ', icn_names{j}];
        idx = idx + 1;
    end
end

figure('Position', [100, 100, 1600, 800]);
violin(reshaped_control_energy', 'xlabel', transition_labels);
title('Distribution of Control Energy for RSN Transitions');
xlabel('RSN Transitions');
ylabel('Control Energy');

xtickangle(90);
set(gca, 'FontSize', 10);

set(gca, 'XTick', 1:(n*m));
set(gca, 'XTickLabel', transition_labels);

set(gcf, 'Color', 'w');

export_fig(gcf,fullfile(outputDir, 'Violin_Plot_Control_Energy.pdf'));


%% CFQ Data 

% Load CFQ data
filename = '/Users/macbookcosima/Desktop/Bachelorarbeit/antrQ_cfq.txt'; 
cfq_data = readtable(filename, 'Delimiter', '\t', 'TreatAsEmpty', {'NA'});

% Sum score 
cfq_items = cfq_data{:, 4:end}; % All CFQ Items
cfq_sum = mean(cfq_items, 2, 'omitnan'); % Sum score as average of CFQ values per person, ignore NA values

% Extract ID, convert to double
cfq_ID = cfq_data.ID;
cfq_ID = cellfun(@(x) str2double(regexprep(x, '\D', '')), cfq_ID);

% Table of CFQ Data (ID and Sum)
cfq_table = table(cfq_ID, cfq_sum, 'VariableNames', {'ID', 'CFQ_Sum'});

% Ensure that both ID columns have same data type and structure
participants_ID = participants(1:num_participants)';

% Table with participant IDs
participants_table = table(participants_ID, 'VariableNames', {'ID'});

% Combine Data
% Find common IDs and extract the relevant rows in cfq_table
[common_IDs, idx_participants, idx_cfq] = intersect(participants_table.ID, cfq_table.ID);
cfq_table_filtered = cfq_table(idx_cfq, :);

% Combine tables based on ID
combined_data = innerjoin(participants_table, cfq_table_filtered, 'Keys', 'ID');

% Remove NaN values from combined data
valid_idx = ~isnan(combined_data.CFQ_Sum);
combined_data = combined_data(valid_idx, :);

disp(combined_data);

%% Descriptive Analysis of CFQ Data (including missings)

% Total number of participants
total_participants = length(cfq_table.CFQ_Sum);  % 88

% CFQ Data (with missings)
cfq_data_all = cfq_table.CFQ_Sum;    

% 1. Average (ignore missings)
mean_cfq = mean(cfq_data_all, 'omitnan');        % 2.15

% 2. Standard Deviation (ignore missings)
sd_cfq = std(cfq_data_all, 'omitnan');           % 0.46

% 3. Median (ignore missings)
median_cfq = median(cfq_data_all, 'omitnan');    % 2.12

% 4. Minimum (ignore missings)
min_cfq = min(cfq_data_all, [], 'omitnan');      % 1.25

% 5. Maximum (ignore missings)
max_cfq = max(cfq_data_all, [], 'omitnan');      % 3.28

% 6. Number of missings
num_missing = sum(isnan(cfq_data_all));          % 7 

% Display results
fprintf('Anzahl aller Personen: %d\n', total_participants);
fprintf('Mittelwert: %.2f\n', mean_cfq);
fprintf('Standardabweichung: %.2f\n', sd_cfq);
fprintf('Median: %.2f\n', median_cfq);
fprintf('Minimum: %.2f\n', min_cfq);
fprintf('Maximum: %.2f\n', max_cfq);
fprintf('Anzahl der Missings: %d\n', num_missing);


%% Analysis of gender and age

% 1. Number of participants by gender (including NaN values)
gender_data = cfq_data.antrQ_sex;

num_gender_1 = sum(gender_data == 1);
num_gender_2 = sum(gender_data == 2);
num_gender_nan = sum(isnan(gender_data));

% 2. Age analysis (including NaN values)
age_data = cfq_data.antrQ_age;

% Remove NaN values from data for analysis
valid_age_data = age_data(~isnan(age_data));

% Age statistics
mean_age = mean(valid_age_data);
sd_age = std(valid_age_data);
min_age = min(valid_age_data);
max_age = max(valid_age_data);

% Number of missing age responses
num_age_nan = sum(isnan(age_data));

% Display results
fprintf('Geschlecht 1 : %d\n', num_gender_1);
fprintf('Geschlecht 2 : %d\n', num_gender_2);
fprintf('Fehlende Geschlechtsangaben: %d\n', num_gender_nan);

fprintf('Durchschnittsalter: %.2f\n', mean_age);
fprintf('Standardabweichung Alter: %.2f\n', sd_age);
fprintf('Niedrigstes Alter: %.2f\n', min_age);
fprintf('Höchstes Alter: %.2f\n', max_age);
fprintf('Fehlende Altersangaben: %d\n', num_age_nan);

% Geschlecht 1 : 40
% Geschlecht 2 : 43
% Fehlende Geschlechtsangaben: 5

% Durchschnittsalter: 26.10
% Standardabweichung Alter: 5.41
% Niedrigstes Alter: 18.00
% Höchstes Alter: 35.00
% Fehlende Altersangaben: 5


%% Calculate Cronbach's Alpha for CFQ Data

% Extract CFQ Items (columns 4 to 35)
cfq_items = cfq_data{:, 4:35};

% Remove rows with missing values (NaN)
valid_cfq_items = cfq_items(~any(isnan(cfq_items), 2), :);

% Calculate Cronbach's Alpha
k = size(valid_cfq_items, 2);
item_variances = var(valid_cfq_items);
total_variance = var(sum(valid_cfq_items, 2));

cronbach_alpha = (k / (k - 1)) * (1 - sum(item_variances) / total_variance);

% Display result
fprintf('Cronbach''s Alpha: %.2f\n', cronbach_alpha);

% Cronbach's Alpha: 0.90


%% Berechnung der deskriptiven Statistiken für die CFQ-Daten (gefilterte Personen nur noch die für die wir auch control energy haben)

% CFQ-Daten (aus der Spalte CFQ_Sum in combined_data)
cfq_data_filtered = combined_data.CFQ_Sum;

% 1. Mittelwert (Missings ignorieren)
mean_cfq_filtered = mean(cfq_data_filtered, 'omitnan');

% 2. Standardabweichung (Missings ignorieren)
sd_cfq_filtered = std(cfq_data_filtered, 'omitnan');

% 3. Median (Missings ignorieren)
median_cfq_filtered = median(cfq_data_filtered, 'omitnan');

% 4. Minimum (Missings ignorieren)
min_cfq_filtered = min(cfq_data_filtered, [], 'omitnan');

% 5. Maximum (Missings ignorieren)
max_cfq_filtered = max(cfq_data_filtered, [], 'omitnan');

% 6. Anzahl der Missings in den gefilterten CFQ-Daten
num_missing_filtered = sum(isnan(cfq_data_filtered));

% Anzahl aller gefilterten Personen
total_participants_filtered = size(cfq_data_filtered, 1);
fprintf('Anzahl aller gefilterten Personen: %d\n', total_participants_filtered);

% CFQ-Statistiken
fprintf('CFQ Mittelwert: %.2f\n', mean_cfq_filtered);
fprintf('CFQ Standardabweichung: %.2f\n', sd_cfq_filtered);
fprintf('CFQ Median: %.2f\n', median_cfq_filtered);
fprintf('CFQ Minimum: %.2f\n', min_cfq_filtered);
fprintf('CFQ Maximum: %.2f\n', max_cfq_filtered);
fprintf('Anzahl der Missings in den gefilterten CFQ-Daten: %d\n', num_missing_filtered);

% Anzahl aller gefilterten Personen: 68
% CFQ Mittelwert: 2.14
% CFQ Standardabweichung: 0.41
% CFQ Median: 2.12
% CFQ Minimum: 1.25
% CFQ Maximum: 3.28
% Anzahl der Missings in den gefilterten CFQ-Daten: 0


% Entfernen von 'CoCo_' und Konvertieren der IDs in numerische Werte
cfq_ID_converted = cellfun(@(x) str2double(regexprep(x, 'CoCo_', '')), cfq_data.ID);


% Alters- und Geschlechtsdaten basierend auf den gefilterten IDs extrahieren

% Altersdaten extrahieren
age_data_filtered = cfq_data{ismember(cfq_ID_converted, combined_data.ID), 'antrQ_age'}; 

% Geschlechtsdaten extrahieren
gender_data_filtered = cfq_data{ismember(cfq_ID_converted, combined_data.ID), 'antrQ_sex'}; 


% Altersanalyse der gefilterten Daten

% Entferne NaN-Werte für die Altersanalyse
valid_age_data_filtered = age_data_filtered(~isnan(age_data_filtered));

% Altersstatistiken berechnen
mean_age_filtered = mean(valid_age_data_filtered);
sd_age_filtered = std(valid_age_data_filtered);
min_age_filtered = min(valid_age_data_filtered);
max_age_filtered = max(valid_age_data_filtered);

% Anzahl der fehlenden Altersangaben in den gefilterten Daten
num_age_nan_filtered = sum(isnan(age_data_filtered));


% Geschlechtsanalyse der gefilterten Daten

% Zähle Anzahl von Geschlecht 1, Geschlecht 2 und NaN in den gefilterten Daten
num_gender_1_filtered = sum(gender_data_filtered == 1);
num_gender_2_filtered = sum(gender_data_filtered == 2);
num_gender_nan_filtered = sum(isnan(gender_data_filtered));


% Berechnung von Cronbach's Alpha für die gefilterten CFQ-Daten
% CFQ-Items extrahieren (angenommen, sie befinden sich in Spalten 4-35 des originalen Datensatzes, aber nur für die gefilterten IDs)
cfq_items_filtered = cfq_data{ismember(cfq_ID_converted, combined_data.ID), 4:35}; % gefilterte CFQ-Items basierend auf IDs

% Entferne Zeilen mit fehlenden Werten (NaN) in den Items
valid_cfq_items_filtered = cfq_items_filtered(~any(isnan(cfq_items_filtered), 2), :);

% Berechnung von Cronbach's Alpha
k_filtered = size(valid_cfq_items_filtered, 2); % Anzahl der Items
item_variances_filtered = var(valid_cfq_items_filtered); % Varianzen der einzelnen Items
total_variance_filtered = var(sum(valid_cfq_items_filtered, 2)); % Varianz der Summenwerte

cronbach_alpha_filtered = (k_filtered / (k_filtered - 1)) * (1 - sum(item_variances_filtered) / total_variance_filtered);

% age
fprintf('Durchschnittsalter der gefilterten Daten: %.2f\n', mean_age_filtered);
fprintf('Standardabweichung Alter: %.2f\n', sd_age_filtered);
fprintf('Niedrigstes Alter: %.2f\n', min_age_filtered);
fprintf('Höchstes Alter: %.2f\n', max_age_filtered);
fprintf('Fehlende Altersangaben in den gefilterten Daten: %d\n', num_age_nan_filtered);

% gender
fprintf('Geschlecht 1: %d\n', num_gender_1_filtered);
fprintf('Geschlecht 2: %d\n', num_gender_2_filtered);
fprintf('Fehlende Geschlechtsangaben: %d\n', num_gender_nan_filtered);

% Cronbach's Alpha
fprintf('Cronbach''s Alpha für die gefilterten Daten: %.2f\n', cronbach_alpha_filtered);


% Anzahl der übereinstimmenden IDs nach Bereinigung: 68
% Durchschnittsalter der gefilterten Daten: 25.99
% Standardabweichung Alter: 5.21
% Niedrigstes Alter: 18.00
% Höchstes Alter: 35.00
% Fehlende Altersangaben in den gefilterten Daten: 0
% Geschlecht 1: 34
% Geschlecht 2: 34
% Fehlende Geschlechtsangaben: 0
% Cronbach's Alpha für die gefilterten Daten: 0.87

%% Deskriptive analyse für die berechnung der control energy (mit den Daten für Geschlecht und Alter aus den CFQ Daten) 
% Deskriptive Analyse für Alters- und Geschlechtsdaten der Personen in der Control Energy-Berechnung

% Falls die IDs in participants als Zellarray vorliegen, in double konvertieren
if iscell(participants)
    participants_IDs = cellfun(@str2double, participants);
else
    participants_IDs = participants;
end

% Anzahl der Personen in participants
total_participants = length(participants_IDs);

% IDs der Personen in cfq_data abgleichen
[~, idx] = ismember(participants_IDs, cfq_ID_converted);

% Gültige Indizes extrahieren (IDs, die in both participants und cfq_data enthalten sind)
valid_idx = idx(idx > 0);

% Extrahiere Alters- und Geschlechtsdaten für die Teilnehmer
age_data_participants = cfq_data.antrQ_age(valid_idx);
gender_data_participants = cfq_data.antrQ_sex(valid_idx);


% Deskriptive Analyse für Altersdaten

% Entferne NaN-Werte aus den Altersdaten für die Analyse
valid_age_data = age_data_participants(~isnan(age_data_participants));

% Altersstatistiken berechnen
mean_age = mean(valid_age_data);
sd_age = std(valid_age_data);
min_age = min(valid_age_data);
max_age = max(valid_age_data);
median_age = median(valid_age_data);

% Anzahl der fehlenden Altersangaben
num_missing_age = sum(isnan(age_data_participants));


% Deskriptive Analyse für Geschlechtsdaten

% Geschlechtsverteilung (ignoriere NaN)
valid_gender_data = gender_data_participants(~isnan(gender_data_participants));
num_gender_1 = sum(valid_gender_data == 1);
num_gender_2 = sum(valid_gender_data == 2);
num_missing_gender = sum(isnan(gender_data_participants));


fprintf('Anzahl der Personen in participants: %d\n', total_participants);
fprintf('Durchschnittsalter: %.2f\n', mean_age);
fprintf('Standardabweichung Alter: %.2f\n', sd_age);
fprintf('Niedrigstes Alter: %.2f\n', min_age);
fprintf('Höchstes Alter: %.2f\n', max_age);
fprintf('Median Alter: %.2f\n', median_age);
fprintf('Fehlende Altersangaben: %d\n', num_missing_age);

fprintf('Geschlecht 1: %d\n', num_gender_1);
fprintf('Geschlecht 2: %d\n', num_gender_2);
fprintf('Fehlende Geschlechtsangaben: %d\n', num_missing_gender);

% Anzahl der Personen in participants: 73
% Durchschnittsalter: 26.06
% Standardabweichung Alter: 5.21
% Niedrigstes Alter: 18.00
% Höchstes Alter: 35.00
% Median Alter: 26.00
% Fehlende Altersangaben: 4
% Geschlecht 1: 34
% Geschlecht 2: 35
% Fehlende Geschlechtsangaben: 4

%% Deskriptive Analyse für die Verteilung der Control Energy Werte für Transitions involving DMN und oder CON

% Annahme: control_energy ist bereits berechnet worden
% control_energy(icn_idx1, icn_idx2, subj_index)

% Indizes für die RSNs
num_icns = 8; % Anzahl der RSNs (einschließlich CON und DMN)
con_index = 1; % CON
dmn_index = 3; % DMN

% Gesamtanzahl der Teilnehmer
num_participants = size(control_energy, 3);

% Speichere Ergebnisse für alle Transitionen von und zu DMN und CON
all_transitions_dmn_con = [];

% Schleife über alle Teilnehmer
for subj_index = 1:num_participants
    % Alle Transitionen von und zu DMN und CON
    for icn_idx = 1:length(icn_names)
        % Transitionen von CON zu allen anderen RSNs
        all_transitions_dmn_con = [all_transitions_dmn_con; control_energy(find(strcmp(icn_names, 'CON')), icn_idx, subj_index)];
        % Transitionen von DMN zu allen anderen RSNs
        all_transitions_dmn_con = [all_transitions_dmn_con; control_energy(find(strcmp(icn_names, 'DMN')), icn_idx, subj_index)];
        % Transitionen von allen anderen RSNs zu CON
        all_transitions_dmn_con = [all_transitions_dmn_con; control_energy(icn_idx, find(strcmp(icn_names, 'CON')), subj_index)];
        % Transitionen von allen anderen RSNs zu DMN
        all_transitions_dmn_con = [all_transitions_dmn_con; control_energy(icn_idx, find(strcmp(icn_names, 'DMN')), subj_index)];
    end
end

% Berechnungen für alle Transitionen
mean_all = mean(all_transitions_dmn_con);
std_all = std(all_transitions_dmn_con);
min_all = min(all_transitions_dmn_con);
max_all = max(all_transitions_dmn_con);

disp('Mittelwert der Control Energy über alle DMN- und CON-Transitionen:');
disp(mean_all);
disp('Standardabweichung der Control Energy über alle DMN- und CON-Transitionen:');
disp(std_all);
disp('Minimum der Control Energy über alle DMN- und CON-Transitionen:');
disp(min_all);
disp('Maximum der Control Energy über alle DMN- und CON-Transitionen:');
disp(max_all);

% Berechnungen für jede spezifische Transition
specific_results = zeros(length(icn_names) * 2, 4); % [Mittelwert, SD, Min, Max]
transition_names = cell(length(icn_names) * 2, 1); % um Übergangsbezeichnungen zu speichern

% Schleife über alle RSNs
for icn_idx = 1:length(icn_names)
    % Ergebnisse für CON zu anderen RSNs
    specific_results(icn_idx, :) = [
        mean(control_energy(find(strcmp(icn_names, 'CON')), icn_idx, :)), ...
        std(control_energy(find(strcmp(icn_names, 'CON')), icn_idx, :)), ...
        min(control_energy(find(strcmp(icn_names, 'CON')), icn_idx, :)), ...
        max(control_energy(find(strcmp(icn_names, 'CON')), icn_idx, :))
    ];
    transition_names{icn_idx} = ['CON -> ', icn_names{icn_idx}];

    % Ergebnisse für DMN zu anderen RSNs
    specific_results(icn_idx + length(icn_names), :) = [
        mean(control_energy(find(strcmp(icn_names, 'DMN')), icn_idx, :)), ...
        std(control_energy(find(strcmp(icn_names, 'DMN')), icn_idx, :)), ...
        min(control_energy(find(strcmp(icn_names, 'DMN')), icn_idx, :)), ...
        max(control_energy(find(strcmp(icn_names, 'DMN')), icn_idx, :))
    ];
    transition_names{icn_idx + length(icn_names)} = ['DMN -> ', icn_names{icn_idx}];
end

% Schleife über alle RSNs für Transitionen zu CON und DMN
for icn_idx = 1:length(icn_names)
    % Ergebnisse für andere RSNs zu CON
    specific_results(icn_idx + 2 * length(icn_names), :) = [
        mean(control_energy(icn_idx, find(strcmp(icn_names, 'CON')), :)), ...
        std(control_energy(icn_idx, find(strcmp(icn_names, 'CON')), :)), ...
        min(control_energy(icn_idx, find(strcmp(icn_names, 'CON')), :)), ...
        max(control_energy(icn_idx, find(strcmp(icn_names, 'CON')), :))
    ];
    transition_names{icn_idx + 2 * length(icn_names)} = [icn_names{icn_idx}, ' -> CON'];

    % Ergebnisse für andere RSNs zu DMN
    specific_results(icn_idx + 3 * length(icn_names), :) = [
        mean(control_energy(icn_idx, find(strcmp(icn_names, 'DMN')), :)), ...
        std(control_energy(icn_idx, find(strcmp(icn_names, 'DMN')), :)), ...
        min(control_energy(icn_idx, find(strcmp(icn_names, 'DMN')), :)), ...
        max(control_energy(icn_idx, find(strcmp(icn_names, 'DMN')), :))
    ];
    transition_names{icn_idx + 3 * length(icn_names)} = [icn_names{icn_idx}, ' -> DMN'];
end

% Ausgabe der Ergebnisse für spezifische Transitionen
for i = 1:length(transition_names)
    disp(['Ergebnisse für die Transition ', transition_names{i}, ':']);
    disp(['Mittelwert: ', num2str(specific_results(i, 1))]);
    disp(['Standardabweichung: ', num2str(specific_results(i, 2))]);
    disp(['Minimum: ', num2str(specific_results(i, 3))]);
    disp(['Maximum: ', num2str(specific_results(i, 4))]);
end


%% Boxplots für die Control Energy erstellen für DMN und CON transitions

figure;

% 1. Subplot: DMN -> alle RSNs
subplot(2, 2, 1);
dmn_to_all = squeeze(control_energy(find(strcmp(icn_names, 'DMN')), :, :)); % DMN zu allen RSNs
boxplot(dmn_to_all', 'Labels', icn_names); % Transponieren, um die richtige Form zu erhalten
title('DMN -> All RSNs');
ylabel('Control Energy');
xlabel('RSNs');

% 2. Subplot: alle RSNs -> DMN
subplot(2, 2, 2);
all_to_dmn = squeeze(control_energy(:, find(strcmp(icn_names, 'DMN')), :)); % Alle RSNs zu DMN
boxplot(all_to_dmn', 'Labels', icn_names); % Transponieren
title('All RSNs -> DMN');
ylabel('Control Energy');
xlabel('RSNs');

% 3. Subplot: CON -> alle RSNs
subplot(2, 2, 3);
con_to_all = squeeze(control_energy(find(strcmp(icn_names, 'CON')), :, :)); % CON zu allen RSNs
boxplot(con_to_all', 'Labels', icn_names); % Transponieren
title('CON -> All RSNs');
ylabel('Control Energy');
xlabel('RSNs');

% 4. Subplot: alle RSNs -> CON
subplot(2, 2, 4);
all_to_con = squeeze(control_energy(:, find(strcmp(icn_names, 'CON')), :)); % Alle RSNs zu CON
boxplot(all_to_con', 'Labels', icn_names); % Transponieren
title('All RSNs -> CON');
ylabel('Control Energy');
xlabel('RSNs');

% Set background color to white
set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'control_energy_dmn_con_analysis.pdf'));


%% Analyse: welche Personen sind durch welchen Prozess herausgeflogen?

% CFQ-Summen berechnen

% CFQ-Items extrahieren (Spalten 4 bis 35)
cfq_items = cfq_data{:, 4:35};

% CFQ-Summe berechnen (als Mittelwert über die CFQ-Items, ignorieren von NaN-Werten)
cfq_sum = mean(cfq_items, 2, 'omitnan');


% IDs der Personen mit fehlenden CFQ-Daten ermitteln
cfq_missing_idx = isnan(cfq_sum);  % Personen mit fehlenden CFQ-Daten
cfq_missing_IDs = cfq_ID_converted(cfq_missing_idx);


% IDs der Personen mit fehlenden Control Energy-Daten ermitteln

% IDs der Personen in participants, die für Control Energy verwendet wurden
participants_IDs = participants;  % Annahme: 'participants' enthält die IDs der verwendeten Personen

% IDs der Personen, die für Control Energy ausgeschlossen wurden (nicht in participants enthalten)
control_energy_excluded_IDs = setdiff(cfq_ID_converted, participants_IDs);


% Vergleich der fehlenden Personen in CFQ und Control Energy

% Personen, die nur bei CFQ herausgefallen sind
only_cfq_missing_IDs = setdiff(cfq_missing_IDs, control_energy_excluded_IDs);

% Personen, die nur bei Control Energy herausgefallen sind
only_control_energy_missing_IDs = setdiff(control_energy_excluded_IDs, cfq_missing_IDs);

% Personen, die sowohl bei CFQ als auch bei Control Energy herausgefallen sind
both_missing_IDs = intersect(cfq_missing_IDs, control_energy_excluded_IDs);


% Anzahl der Personen, die nur bei CFQ fehlten
num_only_cfq_missing = length(only_cfq_missing_IDs);
fprintf('Anzahl der Personen, die nur bei CFQ fehlten: %d\n', num_only_cfq_missing);

% Anzahl der Personen, die nur bei Control Energy fehlten
num_only_control_energy_missing = length(only_control_energy_missing_IDs);
fprintf('Anzahl der Personen, die nur bei Control Energy fehlten: %d\n', num_only_control_energy_missing);

% Anzahl der Personen, die sowohl bei CFQ als auch bei Control Energy fehlten
num_both_missing = length(both_missing_IDs);
fprintf('Anzahl der Personen, die sowohl bei CFQ als auch bei Control Energy fehlten: %d\n', num_both_missing);


% Anzahl der Personen, die nur bei CFQ fehlten: 5
% Anzahl der Personen, die nur bei Control Energy fehlten: 13
% Anzahl der Personen, die sowohl bei CFQ als auch bei Control Energy fehlten: 2



%% Überprüfung der Annahme der Normalverteilung Kolmogorov-Smirnov-Test (KS-Test)

% 1. KS-Test für CFQ-Daten

cfq_sum_data = combined_data.CFQ_Sum;  

% KS-Test auf Normalverteilung für die CFQ-Daten
[h_cfq, p_cfq] = kstest(cfq_sum_data, 'CDF', makedist('Normal'));


if h_cfq == 0
    disp('Die CFQ-Daten folgen der Normalverteilung (Nullhypothese kann nicht verworfen werden).');
else
    disp('Die CFQ-Daten folgen nicht der Normalverteilung (Nullhypothese wird verworfen).');
end
disp(['p-Wert für CFQ-Daten: ', num2str(p_cfq)]);


% Die CFQ-Daten folgen nicht der Normalverteilung (Nullhypothese wird verworfen).
% p-Wert für CFQ-Daten: 8.3519e-50

% Histogramm Verteilung der CFQ Scores
figure;

histogram(combined_data.CFQ_Sum, 30);  % CFQ Scores der kombinierten Daten
title('Distribution of CFQ Scores');
xlabel('CFQ Score');
ylabel('Frequency');

set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'CFQ_Score_Distribution.pdf'));


%% 2. KS-Test für Control Energy-Daten (nur für Transitions mit CON und DMN)

% Relevante ICN-Transitions: CON und DMN
con_idx = find(strcmp(icn_names, 'CON'));
dmn_idx = find(strcmp(icn_names, 'DMN'));
relevant_icns = [con_idx, dmn_idx];


control_energy_dmn_con = [];

for i = 1:num_icns
    for j = 1:length(relevant_icns)
        icn_idx1 = relevant_icns(j);
        
        control_energy_dmn_con = [control_energy_dmn_con; squeeze(control_energy(icn_idx1, i, :))];
        control_energy_dmn_con = [control_energy_dmn_con; squeeze(control_energy(i, icn_idx1, :))];
    end
end

% threshold für sehr kleine p-Werte
min_p_value_threshold = 1e-50;  

% KS-Test auf Normalverteilung für die Control Energy-Daten (nur DMN und CON Transitions)
[h_ce, p_ce] = kstest(control_energy_dmn_con, 'CDF', makedist('Normal'));


if h_ce == 0
    disp('Die Control Energy-Daten für DMN und CON folgen der Normalverteilung (Nullhypothese kann nicht verworfen werden).');
else
    disp('Die Control Energy-Daten für DMN und CON folgen nicht der Normalverteilung (Nullhypothese wird verworfen).');
end

% Überprüfen, ob p-Wert extrem klein ist oder als 0 ausgegeben wird
if p_ce == 0 || p_ce < min_p_value_threshold
    disp(['p-Wert für Control Energy-Daten (DMN und CON Transitions): < ', num2str(min_p_value_threshold)]);
else
    disp(['p-Wert für Control Energy-Daten (DMN und CON Transitions): ', num2str(p_ce, '%.16e')]);
end

% Die Control Energy-Daten für DMN und CON folgen nicht der Normalverteilung (Nullhypothese wird verworfen).
% p-Wert für Control Energy-Daten (DMN und CON Transitions): 0
% p-Wert für Control Energy-Daten (DMN und CON Transitions): < 1e-50


%% Relevante ICN-Transitions: CON und DMN
con_idx = find(strcmp(icn_names, 'CON'));
dmn_idx = find(strcmp(icn_names, 'DMN'));
relevant_icns = [con_idx, dmn_idx];


% Subplot figure
figure;

% 1. Subplot: DMN -> all other RSNs
subplot(2, 2, 1);
control_energy_dmn_to_others = [];
for i = 1:num_icns
    if i ~= dmn_idx
        control_energy_dmn_to_others = [control_energy_dmn_to_others; squeeze(control_energy(dmn_idx, i, :))];
    end
end
histogram(control_energy_dmn_to_others, 30);
title('DMN -> all other RSNs');
xlabel('Control Energy');
ylabel('Frequency');
ylim([0 205]); % Set y-axis limit
%xlim([0 4.3]); % Set x-axis limit

% 2. Subplot: all other RSNs -> DMN
subplot(2, 2, 2);
control_energy_others_to_dmn = [];
for i = 1:num_icns
    if i ~= dmn_idx
        control_energy_others_to_dmn = [control_energy_others_to_dmn; squeeze(control_energy(i, dmn_idx, :))];
    end
end
histogram(control_energy_others_to_dmn, 30);
title('All other RSNs -> DMN');
xlabel('Control Energy');
ylabel('Frequency');
ylim([0 205]); % Set y-axis limit
%xlim([0 4.3]); % Set x-axis limit

% 3. Subplot: CON -> all other RSNs
subplot(2, 2, 3);
control_energy_con_to_others = [];
for i = 1:num_icns
    if i ~= con_idx
        control_energy_con_to_others = [control_energy_con_to_others; squeeze(control_energy(con_idx, i, :))];
    end
end
histogram(control_energy_con_to_others, 30);
title('CON -> all other RSNs');
xlabel('Control Energy');
ylabel('Frequency');
ylim([0 205]); % Set y-axis limit
%xlim([0 4.3]); % Set x-axis limit

% 4. Subplot: all other RSNs -> CON
subplot(2, 2, 4);
control_energy_others_to_con = [];
for i = 1:num_icns
    if i ~= con_idx
        control_energy_others_to_con = [control_energy_others_to_con; squeeze(control_energy(i, con_idx, :))];
    end
end
histogram(control_energy_others_to_con, 30);
title('All other RSNs -> CON');
xlabel('Control Energy');
ylabel('Frequency');
ylim([0 205]); % Set y-axis limit
%xlim([0 4.3]); % Set x-axis limit

set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'Control_Energy_Distributions_DMN_CON_Transitions.pdf'));


%% Berechnung der Spearman-Rangkorrelationen für Transitions involving DMN und CON

% Relevante ICN-Transitions: DMN und CON
con_idx = find(strcmp(icn_names, 'CON'));
dmn_idx = find(strcmp(icn_names, 'DMN'));
relevant_icns = [con_idx, dmn_idx];  % Nur DMN und CON Transitions

num_icns = length(icn_names);  % Anzahl der ICNs

% Initialisieren von Matrizen für Spearman-Korrelationen und p-Werte
correlations_spearman = NaN(num_icns, num_icns);
p_values_spearman = NaN(num_icns, num_icns);

% Korrelationen für jede Transition involving DMN und CON
for i = 1:num_icns
    for j = 1:length(relevant_icns)
        icn_idx1 = relevant_icns(j);
        
        % Extrahiere die Control Energy-Werte für Transitionen von DMN/CON zu anderen Netzwerken
        control_energy_vector = squeeze(control_energy(icn_idx1, i, :));
        
        % Überprüfen, welche Teilnehmer IDs sowohl in participants_table als auch in combined_data vorhanden sind
        [~, idx] = ismember(participants_table.ID, combined_data.ID);
        valid_idx = idx(idx > 0);
        valid_idx = valid_idx(valid_idx <= length(control_energy_vector) & valid_idx <= height(combined_data));
        
        % Gültige Control Energy-Werte und CFQ Scores extrahieren
        valid_control_energy_vector = control_energy_vector(valid_idx);
        valid_cfqs = combined_data.CFQ_Sum(valid_idx);
        
        % Spearman-Rangkorrelation berechnen
        [r_spearman, p_spearman] = corr(valid_control_energy_vector, valid_cfqs, 'Type', 'Spearman');
        correlations_spearman(icn_idx1, i) = r_spearman;
        p_values_spearman(icn_idx1, i) = p_spearman;
        
        % Extrahiere die Control Energy-Werte für Transitionen von anderen Netzwerken zu DMN/CON
        control_energy_vector = squeeze(control_energy(i, icn_idx1, :));
        valid_control_energy_vector = control_energy_vector(valid_idx);
        valid_cfqs = combined_data.CFQ_Sum(valid_idx);
        
        % Spearman-Rangkorrelation berechnen
        [r_spearman, p_spearman] = corr(valid_control_energy_vector, valid_cfqs, 'Type', 'Spearman');
        correlations_spearman(i, icn_idx1) = r_spearman;
        p_values_spearman(i, icn_idx1) = p_spearman;
    end
end

% Ergebnisse für Spearman-Rangkorrelation anzeigen
disp('Spearman correlation matrix:');
disp(correlations_spearman);
disp('Spearman p-value matrix:');
disp(p_values_spearman);

% Visualisierung der Spearman-Korrelationsmatrix als Heatmap mit Zahlen und dynamischer Textfarbe (Farbscala "jet")

figure;
heatmap_data_spearman = correlations_spearman; % Die Spearman-Korrelationsdaten

% Erstelle die Heatmap
imagesc(heatmap_data_spearman, [-0.20 0.20]);  % Setze den Farbbereich von -0.20 bis 0.20 für die Farbskala
colormap([1 1 1; jet(256)]);
colorbar;
title('Spearman Correlation between Control Energy and CFQ Score');
xlabel('Target RSN');
ylabel('Initial RSN');
set(gca, 'XTick', 1:num_icns, 'XTickLabel', icn_names, 'YTick', 1:num_icns, 'YTickLabel', icn_names);
xtickangle(45);

% Dynamische Textfarben basierend auf den Korrelationswerten
for i = 1:num_icns
    for j = 1:num_icns
        correlation_value = heatmap_data_spearman(i, j);

        % Nur Werte, die nicht NaN sind, anzeigen
        if ~isnan(correlation_value)

            % Dynamische Textfarbe: Ab -0.10 wird die Farbe weiß, bei positiven Werten ab 0.10 ebenfalls
            if correlation_value <= -0.10 || correlation_value >= 0.10
                text_color = 'white';
            else
                text_color = 'black';
            end
        
            % Korrelationswert anzeigen, einschließlich negativer Zahlen
            text(j, i, sprintf('%.2f', correlation_value), 'HorizontalAlignment', 'center', 'Color', text_color, 'FontSize', 10);
        end 
    end
end

set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'control_energy_spearman_correlations_dmn_con_with_labels_jet.pdf'));

% Visualisierung der signifikanten Spearman-Korrelationen als Heatmap (Farbscala "jet")

% Signifikanzniveau
alpha = 0.05;

% Bestimmen der signifikanten Korrelationen ohne Korrektur (Spearman)
significant_spearman = p_values_spearman < alpha;


%% Visualisierung der p-Werte als Heatmap mit Zahlen und dynamischer Textfarbe

figure;
heatmap_data_pvalues = p_values_spearman; % Die p-Wert-Daten

% Erstelle die Heatmap
imagesc(heatmap_data_pvalues, [0.02 0.7]);  % Setze den Farbbereich von 0 bis 0.05 für die Farbskala
colormap([1 1 1; jet(256)]);
colorbar;
title('p-values for Spearman Correlation between Control Energy and CFQ Score');
xlabel('Target RSN');
ylabel('Initial RSN');
set(gca, 'XTick', 1:num_icns, 'XTickLabel', icn_names, 'YTick', 1:num_icns, 'YTickLabel', icn_names);
xtickangle(45);

% Dynamische Textfarben basierend auf den p-Werten
for i = 1:num_icns
    for j = 1:num_icns
        p_value = heatmap_data_pvalues(i, j);

        % Nur Werte, die nicht NaN sind, anzeigen
        if ~isnan(p_value)
            % Dynamische Textfarbe: 
            if p_value < 0.18 || p_value > 0.7
                text_color = 'white';
            else
                text_color = 'black';
            end
            
            % p-Wert anzeigen, mit zwei Nachkommastellen
            text(j, i, sprintf('%.3f', p_value), 'HorizontalAlignment', 'center', 'Color', text_color, 'FontSize', 10);
        end 
    end
end

set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'control_energy_pvalues_dmn_con_with_labels_jet.pdf'));


%% Confidence intervals 
% Initialisieren von Matrizen für die Konfidenzintervalle
ci_lower = NaN(num_icns, num_icns);
ci_upper = NaN(num_icns, num_icns);

% Bootstrap-Konfidenzintervalle für die vorhandenen Spearman-Korrelationen berechnen
for j = 1:length(relevant_icns)
    icn_idx1 = relevant_icns(j);

    for i = 1:num_icns
        % Gültige Control Energy-Werte und CFQ Scores von DMN/CON zu anderen Netzwerken extrahieren
        control_energy_vector = squeeze(control_energy(icn_idx1, i, :));
        [~, idx] = ismember(participants_table.ID, combined_data.ID);
        valid_idx = idx(idx > 0);
        valid_idx = valid_idx(valid_idx <= length(control_energy_vector) & valid_idx <= height(combined_data));

        valid_control_energy_vector = control_energy_vector(valid_idx);
        valid_cfqs = combined_data.CFQ_Sum(valid_idx);

        % Bootstrap für das Konfidenzintervall berechnen
        bootstat = bootstrp(1000, @(x, y) corr(x, y, 'Type', 'Spearman'), valid_control_energy_vector, valid_cfqs);
        ci_lower(icn_idx1, i) = prctile(bootstat, 2.5);
        ci_upper(icn_idx1, i) = prctile(bootstat, 97.5);

        % Für Transitionen von anderen Netzwerken zu DMN/CON
        control_energy_vector = squeeze(control_energy(i, icn_idx1, :));
        valid_control_energy_vector = control_energy_vector(valid_idx);
        
        % Bootstrap für das Konfidenzintervall berechnen
        bootstat = bootstrp(1000, @(x, y) corr(x, y, 'Type', 'Spearman'), valid_control_energy_vector, valid_cfqs);
        ci_lower(i, icn_idx1) = prctile(bootstat, 2.5);
        ci_upper(i, icn_idx1) = prctile(bootstat, 97.5);
    end
end

disp('95% Confidence Intervals - Lower bounds:');
disp(ci_lower);
disp('95% Confidence Intervals - Upper bounds:');
disp(ci_upper);

% Plot für die unteren Grenzen der Konfidenzintervalle mit Zahlen
figure;
imagesc(ci_lower);
colorbar;
title('95% Confidence Intervals - Lower bounds');
xlabel('ICN Index');
ylabel('ICN Index');
set(gca, 'XTick', 1:num_icns, 'YTick', 1:num_icns);
colormap('jet'); 
caxis([-1 1]); 

% Werte in die Heatmap einfügen
for i = 1:num_icns
    for j = 1:num_icns
        if ~isnan(ci_lower(i, j))
            text(j, i, sprintf('%.2f', ci_lower(i, j)), ...
                'HorizontalAlignment', 'center', 'Color', 'black');
        end
    end
end

% Plot für die oberen Grenzen der Konfidenzintervalle mit Zahlen
figure;
imagesc(ci_upper);
colorbar;
title('95% Confidence Intervals - Upper bounds');
xlabel('ICN Index');
ylabel('ICN Index');
set(gca, 'XTick', 1:num_icns, 'YTick', 1:num_icns);
colormap('jet'); 
caxis([-1 1]); 

% Werte in die Heatmap einfügen
for i = 1:num_icns
    for j = 1:num_icns
        if ~isnan(ci_upper(i, j))
            text(j, i, sprintf('%.2f', ci_upper(i, j)), ...
                'HorizontalAlignment', 'center', 'Color', 'black');
        end
    end
end

% Error-Bar-Plot für alle relevanten Transitions involving DMN und CON

% Initialisierung der Listen für Korrelationen, Fehlergrenzen und Labels
correlation_values = [];
ci_lower_bounds = [];
ci_upper_bounds = [];
transition_labels = {};

% Für jede relevante Transition die Korrelation und Konfidenzintervalle sammeln
for j = 1:length(relevant_icns)
    icn_idx1 = relevant_icns(j);
    
    for i = 1:num_icns
        % Nur weiter, falls eine Korrelation berechnet wurde (kein NaN-Wert)
        if ~isnan(correlations_spearman(icn_idx1, i))
            correlation_values = [correlation_values; correlations_spearman(icn_idx1, i)];
            ci_lower_bounds = [ci_lower_bounds; ci_lower(icn_idx1, i)];
            ci_upper_bounds = [ci_upper_bounds; ci_upper(icn_idx1, i)];
            transition_labels = [transition_labels; sprintf('%s to %s', icn_names{icn_idx1}, icn_names{i})];
        end
        
        % Umgekehrte Transition (von i zu icn_idx1)
        if ~isnan(correlations_spearman(i, icn_idx1))
            correlation_values = [correlation_values; correlations_spearman(i, icn_idx1)];
            ci_lower_bounds = [ci_lower_bounds; ci_lower(i, icn_idx1)];
            ci_upper_bounds = [ci_upper_bounds; ci_upper(i, icn_idx1)];
            transition_labels = [transition_labels; sprintf('%s to %s', icn_names{i}, icn_names{icn_idx1})];
        end
    end
end

% Berechnung der Fehlergrenzen für die Error Bars
ci_errors = [correlation_values - ci_lower_bounds, ci_upper_bounds - correlation_values];

% Error-Bar-Plot
figure;
hold on;
errorbar(1:length(correlation_values), correlation_values, ci_errors(:,1), ci_errors(:,2), 'o');
xlim([0, length(correlation_values) + 1]);
ylim([-1, 1]);
xticks(1:length(correlation_values));
xticklabels(transition_labels);
xtickangle(45); % Kippt die Labels zur besseren Lesbarkeit
xlabel('ICN Transitions');
ylabel('Spearman Correlation');
title('Spearman Correlations with 95% Confidence Intervals for Transitions involving CON and DMN');
grid on;

% Schriftgröße der x-Achsenbeschriftungen 
ax = gca;
ax.XAxis.FontSize = 8; 

hold off;

set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'spearman_correlations_with_confidence_intervals_dmn_con_transitions.pdf'));

%% Signifikante Spearman Korrelationen für DMN and CON Transitions

figure;
imagesc(significant_spearman);
colormap([1 1 1; jet(256)]);  % signifikante Werte farbig, nicht signifikante weiß
colorbar;
title('Significant Spearman Correlations (no correction) Control Energy and CFQ Score (DMN & CON)');
xlabel('Target RSN');
ylabel('Initial RSN');
set(gca, 'XTick', 1:num_icns, 'XTickLabel', icn_names, 'YTick', 1:num_icns, 'YTickLabel', icn_names);
xtickangle(45);
set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'control_energy_significant_spearman_correlations_dmn_con_jet.pdf'));


%% Berechnung der Spearman-Rangkorrelationen für alle ICN-Transitions  

num_icns = length(icn_names);  % Anzahl der ICNs

% Initialisieren von Matrizen für Spearman-Korrelationen und p-Werte
correlations_spearman = zeros(num_icns, num_icns);
p_values_spearman = zeros(num_icns, num_icns);

% Korrelationen für jede ICN-Transition berechnen
for icn_idx1 = 1:num_icns
    for icn_idx2 = 1:num_icns
        % Control Energy-Werte extrahieren
        control_energy_vector = squeeze(control_energy(icn_idx1, icn_idx2, :));
        
        % Überprüfen, welche Teilnehmer IDs sowohl in participants_table als auch in combined_data vorhanden sind
        [~, idx] = ismember(participants_table.ID, combined_data.ID);
        valid_idx = idx(idx > 0);
        valid_idx = valid_idx(valid_idx <= length(control_energy_vector) & valid_idx <= height(combined_data));
        
        % Gültige Control Energy-Werte und CFQ Scores extrahieren
        valid_control_energy_vector = control_energy_vector(valid_idx);
        valid_cfqs = combined_data.CFQ_Sum(valid_idx);
        
        % Spearman-Rangkorrelation berechnen
        [r_spearman, p_spearman] = corr(valid_control_energy_vector, valid_cfqs, 'Type', 'Spearman');
        correlations_spearman(icn_idx1, icn_idx2) = r_spearman;
        p_values_spearman(icn_idx1, icn_idx2) = p_spearman;
    end
end

% Ergebnisse für Spearman-Rangkorrelation anzeigen
disp('Spearman correlation matrix:');
disp(correlations_spearman);
disp('Spearman p-value matrix:');
disp(p_values_spearman);

% Visualisierung der Spearman-Korrelationsmatrix als Heatmap mit Zahlen und dynamischer Textfarbe

figure;
heatmap_data_spearman = correlations_spearman; 

% Heatmap
imagesc(heatmap_data_spearman, [-0.20 0.20]);  
colormap(jet);  
colorbar;
title('Heatmap of Spearman Correlation between Control Energy and CFQ Score');
xlabel('Target RSN');
ylabel('Initial RSN');
set(gca, 'XTick', 1:num_icns, 'XTickLabel', icn_names, 'YTick', 1:num_icns, 'YTickLabel', icn_names);
xtickangle(45);

% Dynamische Textfarben basierend auf Korrelationswerten
for i = 1:num_icns
    for j = 1:num_icns
        correlation_value = heatmap_data_spearman(i, j);
        
        % Dynamische Textfarbe: Ab -0.10 wird die Farbe weiß, bei positiven Werten ab 0.10 ebenfalls
        if correlation_value <= -0.10 || correlation_value >= 0.10
            text_color = 'white';
        else
            text_color = 'black';
        end
        
        % Korrelationswert anzeigen, einschließlich negativer Zahlen
        text(j, i, sprintf('%.2f', correlation_value), 'HorizontalAlignment', 'center', 'Color', text_color, 'FontSize', 10);
    end
end

set(gcf, 'Color', 'w');

export_fig(gcf, fullfile(outputDir, 'control_energy_spearman_correlations_with_labels.pdf'));


% Visualisierung der signifikanten Spearman-Korrelationen als Heatmap

% Signifikanzniveau
alpha = 0.05;

% Bestimmen der signifikanten Korrelationen ohne Korrektur (Spearman)
significant_spearman = p_values_spearman < alpha;

figure;
imagesc(significant_spearman);
colormap([1 1 1; jet(256)]);  % signifikante Werte farbig, nicht signifikante Weiß
colorbar;
title('Significant Spearman Correlations (without correction) between Control Energy and CFQ Score');
xlabel('Target RSN');
ylabel('Initial RSN');
set(gca, 'XTick', 1:num_icns, 'XTickLabel', icn_names, 'YTick', 1:num_icns, 'YTickLabel', icn_names);
xtickangle(45);
set(gcf, 'Color', 'w');

% Speichern der Figur
export_fig(gcf, fullfile(outputDir, 'control_energy_significant_spearman_correlations_without_correction_all_transitions_cfq.pdf'));
