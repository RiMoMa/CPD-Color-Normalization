function Iout = normalizeCPRstain(I, Iref)
%addpath(genpath('C:\Users\daedro\Documents\tmpColor\CPD2'))

useOD=false;

Io=200;


if nargin < 2
    Iref=imread('03.tif');
end

h=size(I,1);
w=size(I,2);

A= double(reshape(Iref,[],3));
B= double(reshape(I,[],3));

ODA = double(-log((A+1)/Io));
ODB = double(-log((B+1)/Io));

a=double(Iref);
b=double(I);



opt.method='affine'; % use rigid registration
opt.viz=0;          % show every iteration
opt.outliers=0;     % do not assume any noise 

opt.normalize=1;    % normalize to unit variance and zero mean before registering (default)
opt.scale=1;        % estimate global scaling too (default)
opt.rot=1;          % estimate strictly rotational matrix (default)
opt.corresp=0;      % do not compute the correspondence vector at the end of registration (default)

opt.max_it=85;     % max number of iterations
opt.tol=1e-8;       % tolerance


idxA=round(linspace(1,size(A,1),5000));
idxB=round(linspace(1,size(B,1),5000));

selPixels = any(A(idxA,:)<225, 2);
selPixelsB = any(B(idxB,:)<225, 2);

idxA=idxA(selPixels);
idxB=idxB(selPixelsB);
%A(idxA,:)

if ~useOD
    [Transform, Correspondence]=cpd_register(A(idxA,:), B(idxB,:),opt);
    if strcmp(opt.method,'rigid')||strcmp(opt.method,'affine')
        Bnew=cpd_transform(B, Transform);
    else
        aux=linspace(1,size(B,1),10000);
        
        for i=1:size(aux,1);
            fprintf('%d de %d\n', i, size(aux,1));
            Bnew(aux(i):aux(i+1)-1,:) = cpd_transform(B(aux(i):aux(i+1)-1,:), Transform);
        end
    end
    Iout = uint8(reshape(Bnew, h, w, 3));
else
    [Transform, Correspondence]=cpd_register(ODA(idxA,:), ODB(idxB,:),opt);
    Bnew=cpd_transform(ODB, Transform);
    Bnew=Io*exp(-Bnew);
    Iout = uint8(reshape(Bnew, h, w, 3));
end

if opt.viz==1
    figure,cpd_plot_iter(A(idxA,:), B(idxB,:)); title('Before');
    figure,cpd_plot_iter(A(idxA,:), Transform.Y);  title('After registering');
end

%Bnew=cpd_transform(B, Transform);
%Iout = uint8(reshape(Bnew, h, w, 3));

%figure; subplot(1,4,1); 
%imshow(uint8(b)); title('Original'); 
%subplot(1,4,2); 
%imshow(uint8(bnorm)), title('Normalized-pointcloud');
%subplot(1,4,3); 
%imshow(normalizeStaining(b)), title('Normalized-macenko');
%subplot(1,4,4); 
%imshow(uint8(bnorm2)), title('Normalized-pointcloud');
end
