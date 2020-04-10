clc; clear; close all;

% membaca file citra dalam folder
image_folder = 'data latih';
filenames = dir(fullfile(image_folder, '*.jpg'));
jumlah_data = numel(filenames);

% menginisialisasi variabel data_latih
data_latih = zeros(jumlah_data,5);

% proses ekstraksi ciri orde satu
for k = 1:jumlah_data
    full_name= fullfile(image_folder, filenames(k).name);
    Img = imread(full_name);
    
    %Ekstrasi Ciri Tekstur
    Img = rgb2gray(Img);
    H = imhist(Img)';
    H = H/sum(H);
    I = [0:255];
    CiriMEAN = I*H'; %mean
    CiriENT = -H*log2(H+eps)'; %entropy
    CiriVAR = (I-CiriMEAN).^2*H'; %variance
    CiriSKEW = (I-CiriMEAN).^3*H'/CiriVAR^1.5; %skewness
    CiriKURT = (I-CiriMEAN).^4*H'/CiriVAR^2-3; %kurtois
    data_latih(k,:) = [CiriMEAN,CiriENT,CiriVAR,CiriSKEW,CiriKURT];
end

% penentuan nilai target untuk masing2 jenis koki
target_latih = zeros(1,jumlah_data);
target_latih(1:5) = 1;    % oranda
target_latih(6:10) = 2;   % ranchu
target_latih(11:15) = 3;  % bubble eyes
target_latih(16:20) = 4;  % panda-moor
target_latih(21:25) = 5;  % butterfly tail
target_latih(26:30) = 6;  % black-moor

% pelatihan menggunakan algoritma multisvm
output = multisvm(data_latih,target_latih,data_latih);

% menghitung nilai akurasi pelatihan
[n,~] = find(target_latih==output');
jumlah_benar = sum(n);
akurasi = jumlah_benar/jumlah_data*100

% menyimpan variabel data_latih dan target_latih
save data_latih data_latih
save target_latih target_latih


clc; clear; close all;

% membaca file citra dalam folder
image_folder = 'data uji';
filenames = dir(fullfile(image_folder, '*.jpg'));
jumlah_data = numel(filenames);

% menginisialisasi variabel data_latih
data_uji = zeros(jumlah_data,5);

% proses ekstraksi ciri orde satu
for k = 1:jumlah_data
    full_name= fullfile(image_folder, filenames(k).name);
    Img = imread(full_name);
    
    %Ekstrasi Ciri Tekstur
    Img = rgb2gray(Img);
    H = imhist(Img)';
    H = H/sum(H);
    I = [0:255];
    CiriMEAN = I*H';
    CiriENT = -H*log2(H+eps)';
    CiriVAR = (I-CiriMEAN).^2*H';
    CiriSKEW = (I-CiriMEAN).^3*H'/CiriVAR^1.5;
    CiriKURT = (I-CiriMEAN).^4*H'/CiriVAR^2-3;
    data_uji(k,:) = [CiriMEAN,CiriENT,CiriVAR,CiriSKEW,CiriKURT];
end

% penentuan nilai target untuk masing-masing jenis ikan
target_uji = zeros(1,jumlah_data);
target_uji(1:2) = 1;	% oranda
target_uji(3:4) = 2;	% ranchu
target_uji(5:6) = 3;	% bubble eyes
target_uji(7:8) = 4;	% panda-moor
target_uji(9:10) = 5;	% butterfly tail
target_uji(11:12) = 6;	% black-moor

% load data_latih dan target_latih hasil pelatihan
load data_latih
load target_latih

% pengujian menggunakan algoritma multisvm
output = multisvm(data_latih,target_latih,data_uji);

% menghitung nilai akurasi pengujian
[n,~] = find(target_uji==output');
jumlah_benar = sum(n);
akurasi = jumlah_benar/jumlah_data*100
