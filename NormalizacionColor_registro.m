function NormalizacionColor_registro(Path,OutPath)
%Path='/media/ricardo/bdecafd6-310a-43c1-ab12-59c0202656ee/Datasets/Mitos/20x/*.tiff'
addpath(genpath('./CPD2'))
imRef = imread('testA_35.bmp');

%Path = '/media/ricardo/bdecafd6-310a-43c1-ab12-59c0202656ee/IsbiAutomatico/Ricardo/Ricardo/Data/*/*/*/*.png';
%OutPath = 'DataNormalization_registro/';

%mkdir(OutPath);
listFiles = dir(Path);
%parpool(10)
parfor n=3:length(listFiles)
    ImageName=listFiles(n).name; 
    e=strfind(listFiles(n).folder,'/');
    OutFolder = [OutPath,'/'];
    %mkdir(OutFolder)
    OutputName =  [OutFolder,ImageName];
    fprintf('computing %s\n',OutputName )


    fprintf("imagen interes\n")
    if ~exist(OutputName,'file')
    pathIm = [listFiles(n).folder,'/',listFiles(n).name]
    fprintf("reading: %s\n",pathIm)
    im=imread(pathIm);
    %A= 255-im;
    %B=rgb2hsv(A);
    %C=B;
    %C(:,:,3)=1-B(:,:,3);
    %D=hsv2rgb(C);
    %D=im;
    Iout = normalizeCPRstain(im, imRef);
    imwrite(Iout,OutputName)

%   is_white = sum((im(:) > 230))/length(im(:))>0.80
%   if ~is_white
%    Iout = normalizeCPRstain(im, imRef);
%    imwrite(Iout,OutputName)

   %else
    % imwrite(im,OutputName)

%   end
    else
        fprintf('exist: %s\n',OutputName)
    end

end


delete(gcp('nocreate'))
end

% 
% HSVD =rgb2hsv(Iout);
% hueF =HSVD(:,:,1)<0.5;
% Hpla = HSVD(:,:,1);
% MapaHue=(Hpla.*hueF);
% MapaCambio = hueF.*(1-(MapaHue));
% MapaDejar = not(hueF).*Hpla;
% HSVD(:,:,1) = MapaDejar+MapaCambio;
% RGBN=hsv2rgb(HSVD);
% imshow(Iout)
% figure;imshow(RGBN)
