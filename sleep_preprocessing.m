%% Housekeeping
clear all; close all; clc

%% Load Data

% Load Extracted Data
FileLoc = 'C:\Users\antho\OneDrive\Documents\Calhoun_Lab\Projects\Spectrogram_Explainability\sleep_data\';
file = 'segmented_sc_data.mat';
load([FileLoc,file]);

% Get Subject Numbers
filepath = 'C:\Users\antho\OneDrive\Documents\Calhoun_Lab\Projects\Spectrogram_Explainability\sleep_data\sleep-edf-database-expanded-1.0.0\sleep-cassette\';
PSG_data_files = ls([filepath, '**\*PSG.edf']);
Subj_Num = str2num(PSG_data_files(:,4:5)); % get numerical ID associated with each subject

%% Normalize Data and Combine Across Recordings

step_size = 30;
Fs = 100;

start_rec = 1;
end_rec = 153;

count = 1;
for rec = start_rec:end_rec
    clip = data{1,rec}; 
    data{1,rec} = rec; % extract data from subject and remove it from original cell
    clip_labels = labels{1,rec};
    
    n_clips = floor(length(clip)/(step_size*Fs));
        
    if n_clips ~= length(clip_labels)
        disp('N Clips ~= N Labels')
    end
    
    for j = 1:n_clips
        data2(j,:) = clip((j-1)*step_size*Fs+1:(j-1)*step_size*Fs + step_size*Fs);
    end
    clip = [];
    
    % Normalize Data
    data_mean = mean(data2,'all');
    data_std = std(data2,[],'all');
    
    data2 = (data2-data_mean)./(data_std);
    
    % Combine Data from Multiple Subjects
    if rec == start_rec
        disp(size(data2,1)-length(clip_labels))
        disp(size(data2,1)-n_clips)
               
        X = data2;
        data2 = [];
        Y = clip_labels;
        clip_labels = [];
        subject = Subj_Num(rec)*ones(n_clips,1);        
    else
        disp(size(data2,1)-length(clip_labels))
        disp(size(data2,1)-n_clips)
        
        X = [X;data2];
        data2 = [];
        Y = [Y; clip_labels];
        clip_labels = [];
        subject = [subject; Subj_Num(rec)*ones(n_clips,1)];
    end
    
disp(rec)
end

% Save Data
save('segmented_sc_data.mat','X','Y','subject','-v7.3')
