InputFolder = '/media/ricardo/Datos/SegmentacionGlandulasFinal/Patches1024/';
OutputFolder = '/media/ricardo/Datos/SegmentacionGlandulasFinal/Patches1024Normalized/';

%parpool(11)
FoldersDir = dir(InputFolder);
for n=3:length(FoldersDir)
    caseName = FoldersDir(n).name;
    InputPath = [InputFolder,caseName,'/'];
    OutputPath = [OutputFolder,caseName,'/'];
    mkdir(OutputPath)
    FoldersDir_sub = dir(InputPath);
for ni = 3:length(FoldersDir_sub)
    caseName_sub = FoldersDir_sub(ni).name;
    InputPath_sub = [InputPath,caseName_sub,'/'];
    OutputPath_sub = [OutputPath,caseName_sub,'/'];

    mkdir(OutputPath_sub)
    FoldersDir_sub_sub = dir(InputPath_sub);
for nii = 3:length(FoldersDir_sub_sub)
   try
    caseName_sub_sub = FoldersDir_sub_sub(nii).name;
    InputPath_sub_sub = [InputPath_sub,caseName_sub_sub,'/'];
    OutputPath_sub_sub = [OutputPath_sub,caseName_sub_sub,'/'];

    mkdir(OutputPath_sub_sub)
   % FoldersDir_sub_sub = dir(InputPath_sub_sub);




    NormalizacionColor_registro(InputPath_sub_sub,OutputPath_sub_sub)

   catch
       fprintf('no hay')
   end
end
end
end